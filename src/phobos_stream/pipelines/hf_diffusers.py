"""Real HuggingFace Diffusers 3-stage sequential & block-streamed engine (Wan2.1, FLUX, SDXL)."""

from contextlib import contextmanager
import html
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union
import numpy as np

from phobos_stream.pipeline.stage_runner import StageSequentialRunner, flush_accelerator_memory
from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer
from phobos_stream.types import GenerationConfig, PipelineMetrics


DEFAULT_WAN_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


@contextmanager
def _pre_ampere_bf16_sdpa_fastpath(device_str: str) -> Iterator[None]:
    """Use fused FP16 SDPA on Turing/Volta CUDA (< SM80) when model activations are BF16."""
    import torch
    import torch.nn.functional as F

    use_fastpath = False
    if device_str.startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(torch.device(device_str))
        use_fastpath = major < 8

    if not use_fastpath:
        yield
        return

    orig_sdpa = F.scaled_dot_product_attention

    def _wrapped_sdpa(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        if query.dtype == torch.bfloat16 and query.is_cuda:
            q16 = query.clamp(-65000.0, 65000.0).to(torch.float16)
            k16 = key.clamp(-65000.0, 65000.0).to(torch.float16)
            v16 = value.clamp(-65000.0, 65000.0).to(torch.float16)
            return orig_sdpa(q16, k16, v16, *args, **kwargs).to(torch.bfloat16)
        return orig_sdpa(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = _wrapped_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig_sdpa


def _clean_prompt_text(text: str) -> str:
    cleaned = html.unescape(html.unescape(text.strip()))
    return re.sub(r"\s+", " ", cleaned)


def _resolve_cached_safetensors(model_id_or_path: str, subfolder: str) -> list[Path]:
    """Locate cached `.safetensors` shards for `subfolder` without triggering extra downloads."""
    local_dir = Path(model_id_or_path)
    if local_dir.is_dir():
        return sorted((local_dir / subfolder).glob("*.safetensors"))
    try:
        from huggingface_hub import snapshot_download

        snap_root = Path(
            snapshot_download(
                model_id_or_path,
                allow_patterns=[f"{subfolder}/*.safetensors"],
                local_files_only=True,
            )
        )
        return sorted((snap_root / subfolder).glob("*.safetensors"))
    except Exception:
        return []


class HFDiffusersStreamEngine:
    """Executes real HuggingFace Diffusers video/image pipelines under strict VRAM budgets."""

    def __init__(
        self,
        model_id_or_path: Union[str, Path],
        device: str = "auto",
        stream_mode: BlockStreamMode = BlockStreamMode.HYBRID_BUDGET,
        vram_budget_mb: Optional[float] = None,
        ram_budget_mb: Optional[float] = None,
        torch_dtype_name: str = "bfloat16",
    ) -> None:
        import torch

        self.model_id = str(model_id_or_path)
        self.stream_mode = BlockStreamMode(stream_mode)
        self.vram_budget_mb = vram_budget_mb
        self.ram_budget_mb = ram_budget_mb
        self.last_tier_summary: Dict[str, int] = {}

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if torch_dtype_name == "float16":
            self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        elif torch_dtype_name == "bfloat16":
            # Keep bfloat16 on CUDA/MPS so UMT5/T5 encoders never overflow FP16 range to NaN
            self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        else:
            self.dtype = torch.float32

    @property
    def backend_name(self) -> str:
        return f"diffusers:{self.device}"

    def generate(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[np.ndarray, PipelineMetrics]:
        """Run end-to-end 3-stage generation returning `(T, H, W, 3)` uint8 RGB frames."""
        frames, _, metrics = self.generate_with_latents(config, progress_callback=progress_callback)
        return frames, metrics

    def generate_with_latents(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
        """Run 3-stage real Diffusers pipeline returning `(rgb_frames, raw_latents_np, metrics)`."""
        if "wan" in self.model_id.lower():
            return self._run_wan_pipeline(config, progress_callback)
        return self._run_generic_image_pipeline(config, progress_callback)

    def _encode_umt5_prompt(
        self,
        tokenizer: Any,
        text_encoder: Any,
        text: str,
        max_seq_len: int = 512,
    ) -> Any:
        import torch

        cleaned = _clean_prompt_text(text)
        tok = tokenizer(
            [cleaned],
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = tok.input_ids.to(self.device)
        mask = tok.attention_mask.to(self.device)
        seq_len = int(mask[0].gt(0).sum().item())
        with torch.no_grad():
            hidden = text_encoder(ids, attention_mask=mask).last_hidden_state.to(dtype=self.dtype)
            hidden[0, seq_len:] = 0.0
        return hidden.detach().to("cpu")

    def _run_wan_pipeline(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        runner = StageSequentialRunner()
        raw_latents_holder: Dict[str, np.ndarray] = {}

        def _stage1_encode() -> Dict[str, Any]:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).eval()

            te_shards = _resolve_cached_safetensors(self.model_id, "text_encoder")
            with TorchBlockStreamer(
                text_encoder,
                compute_device=self.device,
                mode=BlockStreamMode.CPU_OFFLOAD,
                safetensors_paths=te_shards,
            ):
                pos_emb = self._encode_umt5_prompt(tokenizer, text_encoder, config.prompt)
                neg_emb = None
                if config.guidance_scale > 1.0:
                    neg_text = (config.negative_prompt or "").strip() or DEFAULT_WAN_NEGATIVE_PROMPT
                    neg_emb = self._encode_umt5_prompt(
                        tokenizer,
                        text_encoder,
                        neg_text,
                    )
            del text_encoder, tokenizer
            return {"prompt_embeds": pos_emb, "negative_prompt_embeds": neg_emb}

        def _stage2_denoise(cond: Dict[str, Any]) -> Tuple[Any, int, int]:
            h_aligned = max(128, (config.height // 16) * 16)
            w_aligned = max(128, (config.width // 16) * 16)
            f_aligned = 1 if config.num_frames <= 1 else max(5, ((config.num_frames - 1) // 4) * 4 + 1)
            flow_shift = 3.0 if min(h_aligned, w_aligned) <= 512 else 5.0
            scheduler = UniPCMultistepScheduler.from_pretrained(
                self.model_id,
                subfolder="scheduler",
                flow_shift=flow_shift,
            )
            vae_cfg = AutoencoderKLWan.load_config(self.model_id, subfolder="vae")
            with torch.device("meta"):
                meta_vae = AutoencoderKLWan.from_config(vae_cfg)

            dit_dtype = torch.bfloat16 if self.dtype == torch.bfloat16 else torch.float16
            transformer = WanTransformer3DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                torch_dtype=dit_dtype,
                low_cpu_mem_usage=True,
            ).eval()

            dit_shards = _resolve_cached_safetensors(self.model_id, "transformer")
            with TorchBlockStreamer(
                transformer,
                compute_device=self.device,
                mode=self.stream_mode,
                safetensors_paths=dit_shards,
                vram_budget_mb=self.vram_budget_mb,
                ram_budget_mb=self.ram_budget_mb,
                activation_reserve_mb=1536,
            ) as streamer:
                self.last_tier_summary = streamer.tier_summary
                pipe_dit = WanPipeline(
                    tokenizer=None,
                    text_encoder=None,
                    transformer=transformer,
                    vae=meta_vae,
                    scheduler=scheduler,
                )
                # Keep `pipe_dit._execution_device` pointing at active CUDA/MPS device even when blocks are on CPU!
                pipe_dit._exclude_from_cpu_offload = []
                gen = torch.Generator(device="cpu").manual_seed(int(config.seed))
                pos_dev = cond["prompt_embeds"].to(self.device, dtype=dit_dtype)
                neg_dev = (
                    cond["negative_prompt_embeds"].to(self.device, dtype=dit_dtype)
                    if cond["negative_prompt_embeds"] is not None and config.guidance_scale > 1.0
                    else None
                )
                with torch.no_grad(), _pre_ampere_bf16_sdpa_fastpath(self.device):
                    out = pipe_dit(
                        prompt=None,
                        negative_prompt=None,
                        prompt_embeds=pos_dev,
                        negative_prompt_embeds=neg_dev,
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

            del pipe_dit, transformer, scheduler, meta_vae, pos_dev, neg_dev
            raw_latents_holder["latents"] = latents_cpu.float().numpy()
            return latents_cpu, config.steps, bytes_streamed

        def _stage3_decode(latents_cpu: Any) -> np.ndarray:
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()

            with torch.no_grad():
                lat = latents_cpu.to(self.device, dtype=vae.dtype)
                lat_mean = torch.tensor(vae.config.latents_mean, device=self.device, dtype=vae.dtype).view(
                    1, vae.config.z_dim, 1, 1, 1
                )
                lat_std = 1.0 / torch.tensor(vae.config.latents_std, device=self.device, dtype=vae.dtype).view(
                    1, vae.config.z_dim, 1, 1, 1
                )
                lat = lat / lat_std + lat_mean
                decoded = vae.decode(lat, return_dict=False)[0]
                vid = ((decoded[0].permute(1, 2, 3, 0).clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
                rgb_np = vid.detach().cpu().numpy()
            del vae, decoded, lat
            return rgb_np

        frames, metrics = runner.run(
            encode_fn=_stage1_encode,
            denoise_fn=_stage2_denoise,
            decode_fn=_stage3_decode,
            total_steps=config.steps,
            progress_callback=progress_callback,
        )
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            metrics.peak_vram_mb = float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 * 1024.0)
        flush_accelerator_memory()
        return frames, raw_latents_holder["latents"], metrics

    def _run_generic_image_pipeline(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
        import torch
        from diffusers import AutoPipelineForText2Image

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        if hasattr(pipe, "enable_sequential_cpu_offload") and self.device.startswith("cuda"):
            pipe.enable_sequential_cpu_offload(device=self.device)
        else:
            pipe.to(self.device)
        if hasattr(pipe, "vae") and pipe.vae is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        gen = torch.Generator(device="cpu").manual_seed(int(config.seed))
        runner = StageSequentialRunner()

        def _denoise(_cond: Any) -> Tuple[Any, int, int]:
            out = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt or None,
                height=config.height,
                width=config.width,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=gen,
                output_type="np",
            )
            imgs = np.clip(out.images * 255.0, 0, 255).astype(np.uint8)
            return imgs, config.steps, 0

        frames, metrics = runner.run(
            encode_fn=lambda: None,
            denoise_fn=_denoise,
            decode_fn=lambda arr: arr,
            total_steps=config.steps,
            progress_callback=progress_callback,
        )
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            metrics.peak_vram_mb = float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 * 1024.0)
        del pipe
        flush_accelerator_memory()
        return frames, frames, metrics
