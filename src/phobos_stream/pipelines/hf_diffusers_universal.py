"""Universal 3-stage HuggingFace Diffusers streamer (`FLUX.1`, `LTX-Video`, `HunyuanVideo`, `Wan2.1-I2V`).

Replaces generic `enable_sequential_cpu_offload()` fallbacks with PhobosStream's true 3-stage
architecture + `TorchBlockStreamer` (`HYBRID_BUDGET` / `CPU_OFFLOAD` / `MMAP_STREAM`) +
`FirstBlockResidualCache` + `PhobosAttentionFastpath`, and adds Image-to-Video (`I2V`) conditioning.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import urllib.request
import numpy as np

from phobos_stream.pipeline.stage_runner import StageSequentialRunner, flush_accelerator_memory
from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer
from phobos_stream.types import GenerationConfig, PipelineMetrics


def load_pil_conditioning_image(source: str, width: int, height: int) -> Any:
    """Load an input conditioning image from a local file path, `http(s)` URL, or base64 string."""
    from PIL import Image

    src = source.strip()
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "PhobosStream/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    elif src.startswith("data:image/") and "," in src:
        b64_part = src.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_part))).convert("RGB")
    elif Path(src).exists():
        img = Image.open(src).convert("RGB")
    else:
        raw = base64.b64decode(src)
        img = Image.open(io.BytesIO(raw)).convert("RGB")

    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img


def run_wan_i2v_pipeline(
    engine: Any,
    config: GenerationConfig,
    resolve_shards_fn: Callable[[str, str], list[Path]],
    default_neg_prompt: str,
    progress_callback: Optional[Callable[[int, int, str], None]],
) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
    """Execute 3-stage Image-to-Video (`WanImageToVideoPipeline`) under strict VRAM budgets."""
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanTransformer3DModel
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection, UMT5EncoderModel

    assert config.input_image, "run_wan_i2v_pipeline requires `config.input_image`."
    if torch.cuda.is_available() and engine.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(engine.device)

    h_aligned = max(128, (config.height // 16) * 16)
    w_aligned = max(128, (config.width // 16) * 16)
    f_aligned = max(5, ((max(5, config.num_frames) - 1) // 4) * 4 + 1)
    pil_img = load_pil_conditioning_image(config.input_image, w_aligned, h_aligned)

    runner = StageSequentialRunner()
    raw_latents_holder: Dict[str, np.ndarray] = {}

    def _stage1_encode() -> Dict[str, Any]:
        tokenizer = AutoTokenizer.from_pretrained(engine.model_id, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            engine.model_id, subfolder="text_encoder", torch_dtype=engine.dtype, low_cpu_mem_usage=True
        ).eval()
        te_shards = resolve_shards_fn(engine.model_id, "text_encoder")
        with TorchBlockStreamer(
            text_encoder, compute_device=engine.device, mode=BlockStreamMode.CPU_OFFLOAD, safetensors_paths=te_shards
        ):
            pos_emb = engine._encode_umt5_prompt(tokenizer, text_encoder, config.prompt)
            neg_emb = None
            if config.guidance_scale > 1.0:
                neg_text = (config.negative_prompt or "").strip() or default_neg_prompt
                neg_emb = engine._encode_umt5_prompt(tokenizer, text_encoder, neg_text)
        del text_encoder, tokenizer
        flush_accelerator_memory()

        img_proc = CLIPImageProcessor.from_pretrained(engine.model_id, subfolder="image_processor")
        img_enc = CLIPVisionModelWithProjection.from_pretrained(
            engine.model_id, subfolder="image_encoder", torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(engine.device).eval()
        with torch.no_grad():
            pix = img_proc(images=pil_img, return_tensors="pt").pixel_values.to(engine.device, dtype=torch.float32)
            img_embeds = img_enc(pix, output_hidden_states=True).hidden_states[-2].detach().cpu()
        del img_enc, img_proc, pix
        return {"prompt_embeds": pos_emb, "negative_prompt_embeds": neg_emb, "image_embeds": img_embeds}

    def _stage2_denoise(cond: Dict[str, Any]) -> Tuple[Any, int, int]:
        flow_shift = 3.0 if min(h_aligned, w_aligned) <= 512 else 5.0
        scheduler = UniPCMultistepScheduler.from_pretrained(engine.model_id, subfolder="scheduler", flow_shift=flow_shift)
        # Encode first-frame conditioning latents with real VAE, then free VAE before loading 14B DiT!
        vae_enc = AutoencoderKLWan.from_pretrained(
            engine.model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(engine.device).eval()

        dit_dtype = torch.bfloat16 if engine.dtype == torch.bfloat16 else torch.float16
        transformer = WanTransformer3DModel.from_pretrained(
            engine.model_id, subfolder="transformer", torch_dtype=dit_dtype, low_cpu_mem_usage=True
        ).eval()

        dit_shards = resolve_shards_fn(engine.model_id, "transformer")
        eff_thresh = max(engine.cache_rel_l1_thresh, float(config.cache_rel_l1_thresh))
        with TorchBlockStreamer(
            transformer,
            compute_device=engine.device,
            mode=engine.stream_mode,
            safetensors_paths=dit_shards,
            vram_budget_mb=engine.vram_budget_mb,
            ram_budget_mb=engine.ram_budget_mb,
            activation_reserve_mb=1536,
            cache_rel_l1_thresh=eff_thresh,
            attn_mode=engine.attention_mode,
        ) as streamer:
            engine.last_tier_summary = streamer.tier_summary
            pipe_i2v = WanImageToVideoPipeline(
                tokenizer=None,
                text_encoder=None,
                image_encoder=None,
                image_processor=None,
                transformer=transformer,
                vae=vae_enc,
                scheduler=scheduler,
            )
            pipe_i2v._exclude_from_cpu_offload = []
            gen = torch.Generator(device="cpu").manual_seed(int(config.seed))
            pos_dev = cond["prompt_embeds"].to(engine.device, dtype=dit_dtype)
            neg_dev = (
                cond["negative_prompt_embeds"].to(engine.device, dtype=dit_dtype)
                if cond["negative_prompt_embeds"] is not None and config.guidance_scale > 1.0
                else None
            )
            img_dev = cond["image_embeds"].to(engine.device, dtype=dit_dtype)
            with torch.no_grad():
                out = pipe_i2v(
                    image=pil_img,
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=pos_dev,
                    negative_prompt_embeds=neg_dev,
                    image_embeds=img_dev,
                    height=h_aligned,
                    width=w_aligned,
                    num_frames=f_aligned,
                    num_inference_steps=config.steps,
                    guidance_scale=config.guidance_scale,
                    generator=gen,
                    output_type="latent",
                )
            latents_cpu = out.frames.detach().to("cpu")
            bytes_streamed = streamer.bytes_streamed
            engine.last_skipped_steps = (
                int(streamer.residual_cache.skipped_steps) if streamer.residual_cache is not None else 0
            )

        del pipe_i2v, transformer, vae_enc, scheduler, pos_dev, neg_dev, img_dev
        raw_latents_holder["latents"] = latents_cpu.float().numpy()
        return latents_cpu, config.steps, bytes_streamed

    def _stage3_decode(latents_cpu: Any) -> np.ndarray:
        vae = AutoencoderKLWan.from_pretrained(
            engine.model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(engine.device).eval()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        with torch.no_grad():
            lat = latents_cpu.to(engine.device, dtype=vae.dtype)
            lat_mean = torch.tensor(vae.config.latents_mean, device=engine.device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            lat_std = 1.0 / torch.tensor(vae.config.latents_std, device=engine.device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            decoded = vae.decode(lat / lat_std + lat_mean, return_dict=False)[0]
            vid = ((decoded[0].permute(1, 2, 3, 0).clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
            rgb_np = vid.detach().cpu().numpy()
        del vae, decoded, lat
        return rgb_np

    frames, metrics = runner.run(_stage1_encode, _stage2_denoise, _stage3_decode, total_steps=config.steps, progress_callback=progress_callback)
    metrics.skipped_steps = engine.last_skipped_steps
    if torch.cuda.is_available() and engine.device.startswith("cuda"):
        metrics.peak_vram_mb = float(torch.cuda.max_memory_allocated(engine.device)) / (1024.0 * 1024.0)
    flush_accelerator_memory()
    return frames, raw_latents_holder["latents"], metrics


def run_universal_diffusers_streamed(
    engine: Any,
    config: GenerationConfig,
    resolve_shards_fn: Callable[[str, str], list[Path]],
    progress_callback: Optional[Callable[[int, int, str], None]],
) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
    """Stream ANY HuggingFace Diffusers pipeline (`FLUX.1`, `LTX-Video`, `HunyuanVideo`, `CogVideoX`, `SDXL`)."""
    import inspect
    import json
    import diffusers
    import torch
    from diffusers import DiffusionPipeline

    if torch.cuda.is_available() and engine.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(engine.device)

    pipe = None
    if config.input_image:
        mid_low = engine.model_id.lower()
        idx_file = Path(engine.model_id) / "model_index.json"
        cls_hint = ""
        if idx_file.exists():
            try:
                cls_hint = str(json.loads(idx_file.read_text(encoding="utf-8")).get("_class_name", "")).lower()
            except Exception:
                cls_hint = ""
        tag = f"{mid_low} {cls_hint}"
        i2v_cls_name = None
        if "ltx" in tag:
            i2v_cls_name = "LTXImageToVideoPipeline"
        elif "cogvideo" in tag:
            i2v_cls_name = "CogVideoXImageToVideoPipeline"
        elif "hunyuan" in tag:
            i2v_cls_name = "HunyuanVideoImageToVideoPipeline"
        i2v_cls = getattr(diffusers, i2v_cls_name, None) if i2v_cls_name else None
        if i2v_cls is not None:
            pipe = i2v_cls.from_pretrained(engine.model_id, torch_dtype=engine.dtype, low_cpu_mem_usage=True)
        else:
            img2img_cls = getattr(diffusers, "AutoPipelineForImage2Image", None)
            if img2img_cls is not None and config.num_frames <= 1:
                try:
                    pipe = img2img_cls.from_pretrained(
                        engine.model_id, torch_dtype=engine.dtype, low_cpu_mem_usage=True
                    )
                except Exception:
                    pipe = None
    if pipe is None:
        pipe = DiffusionPipeline.from_pretrained(engine.model_id, torch_dtype=engine.dtype, low_cpu_mem_usage=True)
    backbone_attr = "transformer" if getattr(pipe, "transformer", None) is not None else "unet"
    backbone = getattr(pipe, backbone_attr, None)
    dit_shards = resolve_shards_fn(engine.model_id, backbone_attr)
    eff_thresh = max(engine.cache_rel_l1_thresh, float(config.cache_rel_l1_thresh))

    # Stream heavy text encoders (`T5`, `UMT5`, `Llama`) in Stage 1, then evict to CPU before Stage 2!
    te_streamers: list[TorchBlockStreamer] = []
    te_names_present: list[str] = []
    target_dev = torch.device(engine.device)
    for te_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        te_mod = getattr(pipe, te_name, None)
        if te_mod is not None and isinstance(te_mod, torch.nn.Module):
            te_names_present.append(te_name)
            try:
                s = TorchBlockStreamer(te_mod, compute_device=engine.device, mode=BlockStreamMode.CPU_OFFLOAD).attach()
                if s.blocks:
                    te_streamers.append(s)
                else:
                    s.detach()
            except Exception:
                pass

    def _move_te_non_blocks(dev: Any) -> None:
        streamed_roots = {ts.model: {b for b, _ in ts.blocks} for ts in te_streamers}
        for te_name in te_names_present:
            te_mod = getattr(pipe, te_name, None)
            if te_mod is None:
                continue
            if te_mod not in streamed_roots:
                te_mod.to(dev)
                continue
            blk_set = streamed_roots[te_mod]
            for m_name, sub_m in te_mod.named_modules():
                if any(m_name == b or m_name.startswith(f"{b}.") for b in blk_set):
                    continue
                for k, p in list(sub_m._parameters.items()):
                    if p is not None and p.device != torch.device(dev):
                        sub_m._parameters[k] = torch.nn.Parameter(p.data.to(dev), requires_grad=False)

    if hasattr(pipe, "encode_prompt"):
        orig_encode_prompt = pipe.encode_prompt

        def _staged_encode_prompt(*a: Any, **kw: Any) -> Any:
            _move_te_non_blocks(target_dev)
            try:
                return orig_encode_prompt(*a, **kw)
            finally:
                _move_te_non_blocks("cpu")
                flush_accelerator_memory()

        pipe.encode_prompt = _staged_encode_prompt
    else:
        _move_te_non_blocks(target_dev)

    vae_mod = getattr(pipe, "vae", None)
    if vae_mod is not None:
        if hasattr(vae_mod, "enable_tiling"):
            vae_mod.enable_tiling()
        if hasattr(vae_mod, "enable_slicing"):
            vae_mod.enable_slicing()
        vae_mod.to("cpu")
        for fn_name in ("encode", "decode"):
            if hasattr(vae_mod, fn_name):
                orig_fn = getattr(vae_mod, fn_name)

                def _make_staged_vae(fn: Any) -> Any:
                    def _wrapped(*a: Any, **kw: Any) -> Any:
                        vae_mod.to(target_dev)
                        a_dev = tuple(x.to(target_dev) if isinstance(x, torch.Tensor) else x for x in a)
                        kw_dev = {k: (v.to(target_dev) if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}
                        try:
                            return fn(*a_dev, **kw_dev)
                        finally:
                            vae_mod.to("cpu")
                            flush_accelerator_memory()

                    return _wrapped

                setattr(vae_mod, fn_name, _make_staged_vae(orig_fn))

    pipe._exclude_from_cpu_offload = [*te_names_present, "vae"]
    gen = torch.Generator(device="cpu").manual_seed(int(config.seed))
    call_params = set(inspect.signature(pipe.__call__).parameters.keys())
    call_kwargs: Dict[str, Any] = {
        "prompt": config.prompt,
        "height": config.height,
        "width": config.width,
        "num_inference_steps": config.steps,
        "guidance_scale": config.guidance_scale,
        "generator": gen,
        "output_type": "np",
    }
    if "negative_prompt" in call_params and config.negative_prompt:
        call_kwargs["negative_prompt"] = config.negative_prompt
    if "num_frames" in call_params and config.num_frames > 1:
        call_kwargs["num_frames"] = config.num_frames
    if "image" in call_params and config.input_image:
        call_kwargs["image"] = load_pil_conditioning_image(config.input_image, config.width, config.height)

    runner = StageSequentialRunner()

    def _run_streamed(_cond: Any) -> Tuple[np.ndarray, int, int]:
        if backbone is None:
            out = pipe(**call_kwargs)
            raw = out.frames[0] if hasattr(out, "frames") else out.images
            arr = np.clip(np.asarray(raw) * 255.0, 0, 255).astype(np.uint8)
            return arr, config.steps, 0

        with TorchBlockStreamer(
            backbone,
            compute_device=engine.device,
            mode=engine.stream_mode,
            safetensors_paths=dit_shards,
            vram_budget_mb=engine.vram_budget_mb,
            ram_budget_mb=engine.ram_budget_mb,
            cache_rel_l1_thresh=eff_thresh,
            attn_mode=engine.attention_mode,
        ) as dit_streamer:
            engine.last_tier_summary = dit_streamer.tier_summary
            with torch.no_grad():
                out = pipe(**call_kwargs)
            bytes_s = dit_streamer.bytes_streamed + sum(ts.bytes_streamed for ts in te_streamers)
            engine.last_skipped_steps = (
                int(dit_streamer.residual_cache.skipped_steps) if dit_streamer.residual_cache is not None else 0
            )

        raw = out.frames[0] if hasattr(out, "frames") else out.images
        raw_np = np.asarray(raw)
        if raw_np.dtype != np.uint8:
            raw_np = np.clip(raw_np * 255.0, 0, 255).round().astype(np.uint8)
        if raw_np.ndim == 3:
            raw_np = raw_np[None, ...]
        return raw_np, config.steps, bytes_s

    try:
        frames, metrics = runner.run(lambda: None, _run_streamed, lambda a: a, total_steps=config.steps, progress_callback=progress_callback)
        metrics.skipped_steps = engine.last_skipped_steps
    finally:
        for ts in te_streamers:
            ts.detach(restore_cpu_weights=False)
        del pipe
        if torch.cuda.is_available() and engine.device.startswith("cuda"):
            metrics.peak_vram_mb = float(torch.cuda.max_memory_allocated(engine.device)) / (1024.0 * 1024.0)
        flush_accelerator_memory()

    return frames, frames, metrics
