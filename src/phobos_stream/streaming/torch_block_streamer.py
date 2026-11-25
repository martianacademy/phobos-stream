"""Universal hook-based block streamer for PyTorch Diffusion Transformers & UNets."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from phobos_stream.storage.mmap_loader import MMapTensorLoader


KNOWN_BLOCK_CONTAINERS: Set[str] = {
    "blocks",
    "layers",
    "double_blocks",
    "single_blocks",
    "single_transformer_blocks",
    "transformer_blocks",
    "noise_refiner",
    "context_refiner",
    "joint_blocks",
    "spatial_blocks",
    "temporal_blocks",
    "v_blocks",
}

STRIP_MODEL_PREFIXES: Tuple[str, ...] = (
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "unet.",
    "model.",
)


class BlockStreamMode(str, Enum):
    """Execution mode for transformer block memory management."""

    STAGE_ONLY = "stage_only"
    CPU_OFFLOAD = "cpu_offload"
    MMAP_STREAM = "mmap_stream"


def _resolve_subattr(root: Any, dotted_path: str) -> Tuple[Any, str]:
    """Return `(parent_module, leaf_attr_name)` for a dotted parameter/buffer path."""
    parts = dotted_path.split(".")
    cur = root
    for part in parts[:-1]:
        cur = getattr(cur, part)
    return cur, parts[-1]


def _assign_tensor(root: Any, dotted_path: str, new_tensor: Any, is_param: bool) -> None:
    """Assign a parameter or buffer tensor safely, including on `meta`-device skeletons."""
    import torch.nn as nn

    parent, leaf = _resolve_subattr(root, dotted_path)
    if is_param:
        existing = parent._parameters.get(leaf)
        if existing is None or getattr(existing, "is_meta", False):
            parent._parameters[leaf] = nn.Parameter(new_tensor, requires_grad=False)
        else:
            existing.data = new_tensor
    else:
        existing = parent._buffers.get(leaf)
        if existing is None or getattr(existing, "is_meta", False):
            parent._buffers[leaf] = new_tensor
        else:
            existing.data = new_tensor


def discover_block_modules(
    root_module: Any,
    block_attr_names: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Any]]:
    """Discover ordered repeating transformer/UNet block modules inside a PyTorch model.

    Uses a two-pass scan so inner helper `ModuleList` attributes (such as `to_out`)
    inside an already-discovered block are never misclassified as top-level blocks.
    """
    import torch.nn as nn

    target_names = set(block_attr_names) if block_attr_names else KNOWN_BLOCK_CONTAINERS
    discovered: List[Tuple[str, Any]] = []
    claimed_prefixes: List[str] = []

    def _is_nested(mod_name: str) -> bool:
        return any(mod_name == p or mod_name.startswith(f"{p}.") for p in claimed_prefixes)

    # Pass 1: Known transformer/DiT block ModuleList names
    for name, mod in root_module.named_modules():
        if not name or _is_nested(name):
            continue
        leaf_name = name.split(".")[-1]
        if isinstance(mod, nn.ModuleList) and leaf_name in target_names and len(mod) > 0:
            claimed_prefixes.append(name)
            for idx, child in enumerate(mod):
                discovered.append((f"{name}.{idx}", child))

    if discovered:
        return discovered

    # Pass 2 fallback: Homogeneous ModuleLists of length >= 2
    for name, mod in root_module.named_modules():
        if not name or _is_nested(name):
            continue
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
            child_types = {type(c) for c in mod}
            if len(child_types) == 1:
                claimed_prefixes.append(name)
                for idx, child in enumerate(mod):
                    discovered.append((f"{name}.{idx}", child))

    return discovered


class TorchBlockStreamer:
    """Streams PyTorch DiT/UNet blocks one-by-one from pinned CPU RAM or `.safetensors` mmap.

    Supports both pre-loaded models (`CPU_OFFLOAD`) and zero-RAM `meta`-device skeletons
    (`MMAP_STREAM`), guaranteeing exact numerical equivalence with monolithic execution.
    """

    def __init__(
        self,
        model: Any,
        compute_device: Union[str, Any] = "auto",
        mode: BlockStreamMode = BlockStreamMode.CPU_OFFLOAD,
        safetensors_paths: Optional[Sequence[Union[str, Path]]] = None,
        block_attr_names: Optional[Sequence[str]] = None,
        pin_memory: bool = True,
        prefetch: bool = True,
    ) -> None:
        import torch

        self.model = model
        self._auto_device = str(compute_device) == "auto"
        if self._auto_device:
            if torch.cuda.is_available():
                self.compute_device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.compute_device = torch.device("mps")
            else:
                self.compute_device = torch.device("cpu")
        else:
            self.compute_device = torch.device(compute_device)
        self.mode = BlockStreamMode(mode)
        self.safetensors_paths = [Path(p) for p in (safetensors_paths or [])]
        self.block_attr_names = block_attr_names
        self.pin_memory = pin_memory
        self.prefetch = prefetch

        self.blocks: List[Tuple[str, Any]] = []
        self.bytes_streamed: int = 0
        self.blocks_executed: int = 0

        self._hooks: List[Any] = []
        self._loaders: List[MMapTensorLoader] = []
        self._cpu_shadow: Dict[int, List[Tuple[str, bool, Any]]] = {}
        self._prefetch_cache: Dict[int, Dict[str, Any]] = {}
        self._prefetch_stream: Optional[Any] = None
        self._mmap_map: Dict[int, List[Tuple[str, bool, MMapTensorLoader, str, Any]]] = {}
        self._attached = False

    def _build_safetensors_index(self) -> Dict[str, Tuple[MMapTensorLoader, str]]:
        if not self.safetensors_paths:
            raise ValueError("MMAP_STREAM mode requires at least one `.safetensors` file path.")
        index: Dict[str, Tuple[MMapTensorLoader, str]] = {}
        for path in self.safetensors_paths:
            loader = MMapTensorLoader(path)
            self._loaders.append(loader)
            for raw_key in loader.get_tensor_names():
                index[raw_key] = (loader, raw_key)
                for prefix in STRIP_MODEL_PREFIXES:
                    if raw_key.startswith(prefix):
                        stripped = raw_key[len(prefix) :]
                        index.setdefault(stripped, (loader, raw_key))
        return index

    @staticmethod
    def _lookup_key(
        st_index: Dict[str, Tuple[MMapTensorLoader, str]],
        candidate: str,
    ) -> Optional[Tuple[MMapTensorLoader, str]]:
        resolved = st_index.get(candidate)
        if resolved is not None:
            return resolved
        for prefix in STRIP_MODEL_PREFIXES:
            if candidate.startswith(prefix) and candidate[len(prefix) :] in st_index:
                return st_index[candidate[len(prefix) :]]
        return None

    @staticmethod
    def _iter_module_tensors(mod: Any) -> List[Tuple[str, bool, Any]]:
        """Yield `(rel_name, is_param, tensor)` for all parameters and persistent buffers."""
        non_persistent: Set[str] = set()
        for sub_prefix, sub_mod in mod.named_modules():
            np_set = getattr(sub_mod, "_non_persistent_buffers_set", set())
            for buf_name in np_set:
                full_b = f"{sub_prefix}.{buf_name}" if sub_prefix else buf_name
                non_persistent.add(full_b)

        items: List[Tuple[str, bool, Any]] = [
            (p_name, True, p) for p_name, p in mod.named_parameters(recurse=True)
        ]
        for b_name, buf in mod.named_buffers(recurse=True):
            if b_name not in non_persistent and buf is not None:
                items.append((b_name, False, buf))
        return items

    def attach(self) -> "TorchBlockStreamer":
        """Attach pre/post forward hooks and offload or evict inactive block tensors."""
        if self._attached or self.mode == BlockStreamMode.STAGE_ONLY:
            return self

        self.blocks = discover_block_modules(self.model, self.block_attr_names)
        if not self.blocks:
            raise ValueError("No repeating transformer/UNet block ModuleList found on model.")

        block_prefixes = tuple(f"{pfx}." for pfx, _ in self.blocks)

        if self.mode == BlockStreamMode.CPU_OFFLOAD:
            for name, is_param, tensor_obj in self._iter_module_tensors(self.model):
                if not name.startswith(block_prefixes) and tensor_obj.device != self.compute_device:
                    _assign_tensor(self.model, name, tensor_obj.data.to(self.compute_device), is_param)
            self._setup_cpu_offload()
        elif self.mode == BlockStreamMode.MMAP_STREAM:
            self._setup_mmap_stream(block_prefixes)

        self._attached = True
        return self

    def _setup_cpu_offload(self) -> None:
        import torch

        use_pin = self.pin_memory and self.compute_device.type == "cuda" and torch.cuda.is_available()
        if self.prefetch and self.compute_device.type == "cuda" and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream(device=self.compute_device)

        for idx, (_, blk) in enumerate(self.blocks):
            shadow: List[Tuple[str, bool, Any]] = []
            for rel_name, is_param, tensor_obj in self._iter_module_tensors(blk):
                cpu_t = tensor_obj.data.detach().to("cpu")
                if use_pin and not cpu_t.is_pinned():
                    cpu_t = cpu_t.pin_memory()
                shadow.append((rel_name, is_param, cpu_t))
                _assign_tensor(blk, rel_name, cpu_t, is_param)
            self._cpu_shadow[idx] = shadow
            self._register_cpu_hooks(idx, blk)

    def _prefetch_cpu_block(self, idx: int) -> None:
        import torch

        if idx >= len(self.blocks) or idx in self._prefetch_cache or self._prefetch_stream is None:
            return
        with torch.cuda.stream(self._prefetch_stream):
            self._prefetch_cache[idx] = {
                rel_name: cpu_t.to(self.compute_device, non_blocking=True)
                for rel_name, _, cpu_t in self._cpu_shadow[idx]
            }

    def _resolve_active_device(self, inputs: Any) -> Any:
        """Match the incoming activation tensor's device dynamically when `compute_device='auto'`."""
        import torch

        if not self._auto_device:
            return self.compute_device
        stack = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
        for item in stack:
            if isinstance(item, torch.Tensor):
                return item.device
            if isinstance(item, (tuple, list)):
                for sub in item:
                    if isinstance(sub, torch.Tensor):
                        return sub.device
        return self.compute_device

    def _register_cpu_hooks(self, idx: int, blk: Any) -> None:
        import torch

        def _pre_hook(module: Any, _inputs: Any) -> None:
            target_dev = self._resolve_active_device(_inputs)
            if (
                self._prefetch_stream is not None
                and target_dev == self.compute_device
                and idx in self._prefetch_cache
            ):
                torch.cuda.current_stream(self.compute_device).wait_stream(self._prefetch_stream)
                dev_map = self._prefetch_cache.pop(idx)
            else:
                dev_map = {
                    rel_name: cpu_t.to(target_dev, non_blocking=(target_dev.type == "cuda"))
                    for rel_name, _, cpu_t in self._cpu_shadow[idx]
                }

            for rel_name, is_param, _ in self._cpu_shadow[idx]:
                dev_t = dev_map[rel_name]
                _assign_tensor(module, rel_name, dev_t, is_param)
                self.bytes_streamed += dev_t.numel() * dev_t.element_size()

            if self._prefetch_stream is not None and target_dev == self.compute_device:
                self._prefetch_cpu_block(idx + 1)

        def _post_hook(module: Any, _inputs: Any, _output: Any) -> None:
            for rel_name, is_param, cpu_t in self._cpu_shadow[idx]:
                _assign_tensor(module, rel_name, cpu_t, is_param)
            self.blocks_executed += 1

        self._hooks.append(blk.register_forward_pre_hook(_pre_hook))
        self._hooks.append(blk.register_forward_hook(_post_hook))

    def _setup_mmap_stream(self, block_prefixes: Tuple[str, ...]) -> None:
        import torch

        try:
            st_index = self._build_safetensors_index()

            # Materialize lightweight non-block stem/head layers once onto compute_device
            for full_name, is_param, tensor_obj in self._iter_module_tensors(self.model):
                if full_name.startswith(block_prefixes):
                    continue
                resolved = self._lookup_key(st_index, full_name)
                if resolved is not None:
                    loader, real_key = resolved
                    loaded = loader.read_torch_tensor(
                        real_key,
                        device=str(self.compute_device),
                        target_dtype=tensor_obj.dtype,
                    )
                    _assign_tensor(self.model, full_name, loaded, is_param)
                elif getattr(tensor_obj, "is_meta", False):
                    raise KeyError(f"Missing non-block meta tensor '{full_name}' in safetensors index.")
                elif tensor_obj.device != self.compute_device:
                    _assign_tensor(self.model, full_name, tensor_obj.data.to(self.compute_device), is_param)

            # Index every repeating block and evict its tensors to 0-element placeholders
            for idx, (blk_prefix, blk) in enumerate(self.blocks):
                blk_entries: List[Tuple[str, bool, MMapTensorLoader, str, Any]] = []
                for rel_name, is_param, tensor_obj in self._iter_module_tensors(blk):
                    candidate = f"{blk_prefix}.{rel_name}"
                    resolved = self._lookup_key(st_index, candidate)
                    if resolved is None:
                        raise KeyError(
                            f"Missing tensor '{candidate}' in safetensors files {[p.name for p in self.safetensors_paths]}."
                        )
                    loader, real_key = resolved
                    orig_dtype = tensor_obj.dtype
                    blk_entries.append((rel_name, is_param, loader, real_key, orig_dtype))
                    _assign_tensor(blk, rel_name, torch.empty(0, dtype=orig_dtype, device="cpu"), is_param)
                self._mmap_map[idx] = blk_entries
                self._register_mmap_hooks(idx, blk)
        except Exception:
            for loader in self._loaders:
                loader.close()
            self._loaders.clear()
            raise

    def _register_mmap_hooks(self, idx: int, blk: Any) -> None:
        import torch

        def _pre_hook(module: Any, _inputs: Any) -> None:
            target_dev = self._resolve_active_device(_inputs)
            for rel_name, is_param, loader, real_key, orig_dtype in self._mmap_map[idx]:
                loaded = loader.read_torch_tensor(
                    real_key,
                    device=str(target_dev),
                    target_dtype=orig_dtype,
                )
                _assign_tensor(module, rel_name, loaded, is_param)
                self.bytes_streamed += loader.get_tensor_byte_size(real_key)

        def _post_hook(module: Any, _inputs: Any, _output: Any) -> None:
            for rel_name, is_param, _, _, orig_dtype in self._mmap_map[idx]:
                _assign_tensor(module, rel_name, torch.empty(0, dtype=orig_dtype, device="cpu"), is_param)
            self.blocks_executed += 1

        self._hooks.append(blk.register_forward_pre_hook(_pre_hook))
        self._hooks.append(blk.register_forward_hook(_post_hook))

    def detach(self, restore_cpu_weights: bool = True) -> None:
        """Remove all forward hooks, restore CPU shadow weights if applicable, and close mmaps."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._prefetch_cache.clear()

        if restore_cpu_weights and self._cpu_shadow:
            for idx, (_, blk) in enumerate(self.blocks):
                for rel_name, is_param, cpu_t in self._cpu_shadow.get(idx, []):
                    _assign_tensor(blk, rel_name, cpu_t, is_param)

        self._cpu_shadow.clear()
        self._mmap_map.clear()
        for loader in self._loaders:
            loader.close()
        self._loaders.clear()
        self._attached = False

    def __enter__(self) -> "TorchBlockStreamer":
        return self.attach()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.detach()
