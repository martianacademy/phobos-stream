"""PyTorch module introspection, block discovery, and safe parameter/buffer placement helpers."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from phobos_stream.storage.mmap_loader import MMapTensorLoader


KNOWN_BLOCK_CONTAINERS: Set[str] = {
    "blocks",
    "block",
    "layers",
    "double_blocks",
    "single_blocks",
    "single_transformer_blocks",
    "transformer_blocks",
    "noise_refiner",
    "context_refiner",
    "refiner_blocks",
    "joint_blocks",
    "spatial_blocks",
    "temporal_blocks",
    "v_blocks",
    "encoder_layers",
    "h",
}

STRIP_MODEL_PREFIXES: Tuple[str, ...] = (
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "unet.",
    "encoder.",
    "model.",
)


def resolve_subattr(root: Any, dotted_path: str) -> Tuple[Any, str]:
    """Walk `dotted_path` on `root` and return `(parent_module, leaf_attr_name)`."""
    parts = dotted_path.split(".")
    cur = root
    for part in parts[:-1]:
        cur = getattr(cur, part)
    return cur, parts[-1]


def assign_tensor(root: Any, dotted_path: str, new_tensor: Any, is_param: bool) -> None:
    """Assign a parameter or buffer tensor safely, including on `meta`-device skeletons."""
    import torch.nn as nn

    parent, leaf = resolve_subattr(root, dotted_path)
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


def move_non_persistent_buffers(root: Any, target_device: Any) -> None:
    """Move materialized non-persistent buffers (e.g. Wan2.1/FLUX RoPE caches) to `target_device`."""
    for _, sub_mod in root.named_modules():
        for buf_name in getattr(sub_mod, "_non_persistent_buffers_set", set()):
            buf = sub_mod._buffers.get(buf_name)
            if buf is not None and not getattr(buf, "is_meta", False) and buf.numel() > 0:
                if buf.device != target_device:
                    sub_mod._buffers[buf_name] = buf.to(target_device)


def iter_persistent_module_tensors(mod: Any) -> List[Tuple[str, bool, Any]]:
    """Return `(rel_name, is_param, tensor)` for all persistent parameters and buffers of `mod`."""
    non_persistent: Set[str] = set()
    for sub_prefix, sub_mod in mod.named_modules():
        for buf_name in getattr(sub_mod, "_non_persistent_buffers_set", set()):
            non_persistent.add(f"{sub_prefix}.{buf_name}" if sub_prefix else buf_name)

    items: List[Tuple[str, bool, Any]] = [
        (p_name, True, p) for p_name, p in mod.named_parameters(recurse=True)
    ]
    for b_name, buf in mod.named_buffers(recurse=True):
        if b_name not in non_persistent and buf is not None:
            items.append((b_name, False, buf))
    return items


def build_safetensors_index(
    paths: Sequence[Path],
) -> Tuple[List[MMapTensorLoader], Dict[str, Tuple[MMapTensorLoader, str]]]:
    """Open `.safetensors` files via `MMapTensorLoader` and build a normalized key lookup map."""
    loaders: List[MMapTensorLoader] = []
    index: Dict[str, Tuple[MMapTensorLoader, str]] = {}
    try:
        for path in paths:
            loader = MMapTensorLoader(path)
            loaders.append(loader)
            for raw_key in loader.get_tensor_names():
                index[raw_key] = (loader, raw_key)
                for prefix in STRIP_MODEL_PREFIXES:
                    if raw_key.startswith(prefix):
                        index.setdefault(raw_key[len(prefix) :], (loader, raw_key))
    except Exception:
        for loader in loaders:
            loader.close()
        raise
    return loaders, index


def lookup_safetensors_key(
    st_index: Dict[str, Tuple[MMapTensorLoader, str]],
    candidate: str,
) -> Optional[Tuple[MMapTensorLoader, str]]:
    """Resolve a module tensor name against raw or prefix-stripped `.safetensors` keys."""
    if candidate in st_index:
        return st_index[candidate]
    for prefix in STRIP_MODEL_PREFIXES:
        if candidate.startswith(prefix) and candidate[len(prefix) :] in st_index:
            return st_index[candidate[len(prefix) :]]
    return None


def discover_block_modules(
    root_module: Any,
    block_attr_names: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Any]]:
    """Discover ordered repeating transformer/encoder/UNet block modules inside a PyTorch model."""
    import torch.nn as nn

    target_names = set(block_attr_names) if block_attr_names else KNOWN_BLOCK_CONTAINERS
    discovered: List[Tuple[str, Any]] = []
    claimed_prefixes: List[str] = []

    def _is_nested(mod_name: str) -> bool:
        return any(mod_name == p or mod_name.startswith(f"{p}.") for p in claimed_prefixes)

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

    for name, mod in root_module.named_modules():
        if not name or _is_nested(name):
            continue
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2 and len({type(c) for c in mod}) == 1:
            claimed_prefixes.append(name)
            for idx, child in enumerate(mod):
                discovered.append((f"{name}.{idx}", child))

    return discovered
