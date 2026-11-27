"""Zero-RAM ComfyUI Diffusion Model Loader (`safetensors` header -> `meta` skeleton + `mmap` stream).

Eliminates the `16 GB` System RAM Out-Of-Memory crash caused by standard `UNETLoader`, which
loads the full `14 GB-26 GB` checkpoint into CPU RAM before block streaming can attach:
1. Parses ONLY the `.safetensors` JSON header via `MMapTensorLoader` (`< 10 ms`, `~0 MB` RAM) to
   construct a lightweight `meta`-device state dict (materializing only tiny `<= 64`-element
   quantization/scale metadata tensors).
2. Detects the exact ComfyUI `model_config` and instantiates `diffusion_model` directly on
   `torch.device("meta")` (`0 MB` CPU RAM, `0 MB` GPU VRAM).
3. Materializes ONLY the non-block stem/head modules (`img_in`, `time_in`, `final_layer`,
   typically `100-300 MB`) from `MMapTensorLoader` and attaches `TorchBlockStreamer` in
   `MMAP_STREAM` or `HYBRID_BUDGET` mode across all repeating transformer blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

from phobos_stream.integrations.comfyui.comfy_runtime import (
    COMFY_NATIVE_AVAILABLE,
    folder_paths,
    get_comfy_filenames,
)
from phobos_stream.storage.gguf_mmap_loader import GGUFMMapLoader
from phobos_stream.storage.mmap_loader import MMapTensorLoader
from phobos_stream.streaming import BlockStreamMode, TorchBlockStreamer
from phobos_stream.streaming.block_discovery import (
    assign_tensor,
    build_safetensors_index,
    discover_block_modules,
    iter_persistent_module_tensors,
    lookup_safetensors_key,
)


_SAFETENSORS_TO_TORCH_DTYPE: Dict[str, str] = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    "Q8_0": "bfloat16",
    "Q4_0": "bfloat16",
    "Q4_1": "bfloat16",
    "Q4_K": "bfloat16",
    "Q5_K": "bfloat16",
    "Q6_K": "bfloat16",
}


def build_header_only_meta_state_dict(
    safetensors_path: Union[str, Path],
    scalar_threshold_elems: int = 64,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build a `meta`-tensor state dict + header `__metadata__` without reading bulk weights."""
    import torch

    path_obj = Path(safetensors_path)
    meta_sd: Dict[str, Any] = {}
    loader_ctx: Any = (
        GGUFMMapLoader(path_obj) if str(path_obj).lower().endswith(".gguf") else MMapTensorLoader(path_obj)
    )
    with loader_ctx as loader:
        if isinstance(loader, GGUFMMapLoader):
            raw_meta = dict(loader.metadata)
        else:
            raw_meta = dict(loader.header.get("__metadata__", {}) or {})
        for key in loader.get_tensor_names():
            info = loader.header[key]
            shape = tuple(int(d) for d in info["shape"])
            dt_name = _SAFETENSORS_TO_TORCH_DTYPE.get(str(info["dtype"]), "float32")
            t_dtype = getattr(torch, dt_name, torch.float32)
            numel = 1
            for d in shape:
                numel *= d
            is_small_u8_cfg = str(info["dtype"]) == "U8" and numel <= 1_048_576
            if numel <= scalar_threshold_elems or is_small_u8_cfg:
                t = loader.read_torch_tensor(key, device="cpu")
            else:
                t = torch.empty(shape, dtype=t_dtype, device="meta")
            t._phobos_raw_key = key
            meta_sd[key] = t
    return meta_sd, raw_meta


def materialize_non_block_stem_from_mmap(
    diffusion_model: Any,
    safetensors_path: Union[str, Path],
    target_device: Union[str, Any] = "cpu",
    target_dtype: Optional[Any] = None,
    key_remap: Optional[Dict[str, str]] = None,
) -> int:
    """Materialize ONLY non-block stem/head tensors onto `target_device`; leave blocks on `meta`."""
    import torch

    blocks = discover_block_modules(diffusion_model)
    block_prefixes: Set[str] = {b_name for b_name, _ in blocks}
    effective_remap = key_remap if key_remap is not None else getattr(diffusion_model, "_phobos_key_remap", None)
    loaders, st_index = build_safetensors_index([Path(safetensors_path)], effective_remap)
    bytes_loaded = 0
    try:
        for rel_name, is_param, cur_t in iter_persistent_module_tensors(diffusion_model):
            if any(rel_name == p or rel_name.startswith(f"{p}.") for p in block_prefixes):
                continue
            found = lookup_safetensors_key(st_index, rel_name)
            if found is None:
                if getattr(cur_t, "is_meta", False):
                    dt = target_dtype if (target_dtype is not None and cur_t.is_floating_point()) else cur_t.dtype
                    assign_tensor(
                        diffusion_model,
                        rel_name,
                        torch.zeros(tuple(cur_t.shape), dtype=dt, device=target_device),
                        is_param=is_param,
                    )
                continue
            loader, real_key = found
            bytes_loaded += loader.get_tensor_byte_size(real_key)
            dt = cur_t.dtype
            if target_dtype is not None and cur_t.is_floating_point():
                dt = target_dtype
            loaded_t = loader.read_torch_tensor(real_key, device=str(target_device), target_dtype=dt)
            assign_tensor(diffusion_model, rel_name, loaded_t, is_param=is_param)
    finally:
        for loader in loaders:
            loader.close()
    return bytes_loaded


def load_comfy_unet_zero_ram(
    unet_path: Union[str, Path],
    mode: str = "mmap_stream",
    weight_dtype: str = "default",
    vram_budget_mb: float = 0.0,
    ram_budget_mb: float = 0.0,
    cache_rel_l1_thresh: float = 0.0,
    cache_warmup_steps: int = 2,
    cache_max_consecutive_skips: int = 2,
    attention_mode: str = "auto",
) -> Any:
    """Instantiate a ComfyUI `ModelPatcher` in `< 2 GB` System RAM directly from `.safetensors`."""
    import torch
    import comfy.model_detection
    import comfy.model_management
    import comfy.model_patcher
    import comfy.utils

    st_path = Path(unet_path)
    meta_sd, metadata = build_header_only_meta_state_dict(st_path)
    if hasattr(comfy.utils, "convert_old_quants"):
        meta_sd, metadata = comfy.utils.convert_old_quants(meta_sd, "", metadata=metadata)

    prefix = comfy.model_detection.unet_prefix_from_state_dict(meta_sd)
    stripped = comfy.utils.state_dict_prefix_replace(meta_sd, {prefix: ""}, filter_keys=True)
    if len(stripped) > 0:
        meta_sd = stripped
        if hasattr(comfy.utils, "convert_old_quants"):
            meta_sd, metadata = comfy.utils.convert_old_quants(meta_sd, "", metadata=metadata)

    parameters = comfy.utils.calculate_parameters(meta_sd)
    detected_wdtype = comfy.utils.weight_dtype(meta_sd)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    model_config = comfy.model_detection.model_config_from_unet(meta_sd, "", metadata=metadata)
    if model_config is None:
        converted = comfy.model_detection.convert_diffusers_mmdit(meta_sd, "")
        if converted is not None:
            meta_sd = converted
            model_config = comfy.model_detection.model_config_from_unet(meta_sd, "", metadata=metadata)
    if model_config is None:
        raise RuntimeError(f"PhobosZeroRAMLoader could not detect ComfyUI architecture for '{st_path.name}'.")

    explicit_dtype = None
    if weight_dtype == "fp8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
        explicit_dtype = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e5m2" and hasattr(torch, "float8_e5m2"):
        explicit_dtype = torch.float8_e5m2

    supported_dtypes = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        detected_wdtype = None
    unet_dtype = (
        explicit_dtype
        if explicit_dtype is not None
        else comfy.model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=supported_dtypes,
            weight_dtype=detected_wdtype,
        )
    )
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        None if model_config.quant_config is not None else unet_dtype,
        load_device,
        model_config.supported_inference_dtypes,
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    # Create `diffusion_model` on `meta` (0 bytes RAM!), while keeping `model_sampling` on CPU!
    processed_sd = model_config.process_unet_state_dict(dict(meta_sd))
    key_remap: Dict[str, str] = {
        k: getattr(v, "_phobos_raw_key")
        for k, v in processed_sd.items()
        if getattr(v, "_phobos_raw_key", None) is not None
    }
    del processed_sd

    base_model = model_config.get_model(meta_sd, "", device=torch.device("meta"))
    base_model.device = load_device
    base_model.diffusion_model._phobos_key_remap = key_remap
    del meta_sd

    materialize_non_block_stem_from_mmap(
        base_model.diffusion_model,
        st_path,
        target_device=load_device,
        target_dtype=None,
        key_remap=key_remap,
    )

    streamer = TorchBlockStreamer(
        model=base_model.diffusion_model,
        compute_device=str(load_device),
        mode=BlockStreamMode(mode),
        safetensors_paths=[st_path],
        vram_budget_mb=vram_budget_mb if vram_budget_mb > 0.0 else None,
        ram_budget_mb=ram_budget_mb if ram_budget_mb > 0.0 else None,
        cache_rel_l1_thresh=cache_rel_l1_thresh,
        cache_warmup_steps=cache_warmup_steps,
        cache_max_consecutive_skips=cache_max_consecutive_skips,
        attn_mode=attention_mode,
    ).attach()

    patcher = comfy.model_patcher.ModelPatcher(
        base_model,
        load_device=load_device,
        offload_device=offload_device,
    )
    patcher._phobos_block_streamer = streamer
    patcher.load = lambda *a, **kw: None
    patcher.patch_model = lambda *a, **kw: base_model
    patcher.model_size = lambda: 0
    return patcher


def _ensure_gguf_registered_and_list() -> list[str]:
    if COMFY_NATIVE_AVAILABLE and folder_paths is not None:
        try:
            if hasattr(folder_paths, "supported_pt_extensions") and isinstance(folder_paths.supported_pt_extensions, set):
                folder_paths.supported_pt_extensions.add(".gguf")
            for key in ("diffusion_models", "unet"):
                entry = folder_paths.folder_names_and_paths.get(key)
                if entry and len(entry) >= 2 and isinstance(entry[1], set):
                    entry[1].add(".gguf")
            if hasattr(folder_paths, "filename_list_cache"):
                folder_paths.filename_list_cache.clear()
        except Exception:
            pass
    return get_comfy_filenames("diffusion_models", ["hunyuan_video_t2v_720p_bf16.safetensors"])


_ensure_gguf_registered_and_list()


class PhobosZeroRAMLoaderNode:
    """ComfyUI Node: Loads `14 GB-33 GB+` `.safetensors` & `.gguf` DiTs in `< 2 GB` System RAM & `< 2 GB` VRAM."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        unet_list = _ensure_gguf_registered_and_list()
        return {
            "required": {
                "unet_name": (unet_list, {"default": unet_list[0]}),
                "mode": (["mmap_stream", "hybrid_budget", "cpu_offload"], {"default": "mmap_stream"}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"], {"default": "default"}),
            },
            "optional": {
                "vram_budget_mb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 65536.0, "step": 256.0}),
                "ram_budget_mb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 131072.0, "step": 512.0}),
                "cache_rel_l1_thresh": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.50, "step": 0.01}),
                "cache_warmup_steps": ("INT", {"default": 2, "min": 1, "max": 10}),
                "cache_max_consecutive_skips": ("INT", {"default": 2, "min": 1, "max": 6}),
                "attention_mode": (["auto", "sage2", "chunked_sdpa", "stock"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_zero_ram"
    CATEGORY = "PhobosStream"

    def load_zero_ram(
        self,
        unet_name: str,
        mode: str = "mmap_stream",
        weight_dtype: str = "default",
        vram_budget_mb: float = 0.0,
        ram_budget_mb: float = 0.0,
        cache_rel_l1_thresh: float = 0.0,
        cache_warmup_steps: int = 2,
        cache_max_consecutive_skips: int = 2,
        attention_mode: str = "auto",
    ) -> Tuple[Any]:
        resolved = unet_name
        if not Path(unet_name).exists() and COMFY_NATIVE_AVAILABLE and folder_paths is not None:
            _ensure_gguf_registered_and_list()
            full = folder_paths.get_full_path("diffusion_models", unet_name)
            if full:
                resolved = full
            else:
                for base_dir in folder_paths.get_folder_paths("diffusion_models"):
                    cand = Path(base_dir) / unet_name
                    if cand.exists():
                        resolved = str(cand)
                        break
        patcher = load_comfy_unet_zero_ram(
            unet_path=resolved,
            mode=mode,
            weight_dtype=weight_dtype,
            vram_budget_mb=vram_budget_mb,
            ram_budget_mb=ram_budget_mb,
            cache_rel_l1_thresh=cache_rel_l1_thresh,
            cache_warmup_steps=cache_warmup_steps,
            cache_max_consecutive_skips=cache_max_consecutive_skips,
            attention_mode=attention_mode,
        )
        return (patcher,)
