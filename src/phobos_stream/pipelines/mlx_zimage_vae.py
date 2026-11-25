"""Apple Silicon MLX Strip-VAE weight mapper and decoder for BFL / Z-Image `ae.safetensors`."""

from pathlib import Path
from typing import Any, List, Tuple, Union
import numpy as np


def load_mlx_flux_vae(vae_path: Union[str, Path], low_memory: bool = True) -> Any:
    """Load `ae.safetensors` decoder weights into an `mflux` VAE with optional strip decoding."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from mflux.models.common.vae.block_decoder import VAEBlockDecoder
    from mflux.models.z_image.model.z_image_vae.vae import VAE

    raw_vae = mx.load(str(vae_path))
    mapped: List[Tuple[str, Any]] = []

    def _tr_conv(w: Any) -> Any:
        return mx.transpose(w, (0, 2, 3, 1)) if w.ndim == 4 else w

    for k, v in raw_vae.items():
        if not k.startswith("decoder."):
            continue
        vb = v.astype(mx.bfloat16)
        if k in ("decoder.conv_in.weight", "decoder.conv_out.weight"):
            mapped.append((k.replace(".weight", ".conv.weight"), _tr_conv(vb)))
        elif k in ("decoder.conv_in.bias", "decoder.conv_out.bias"):
            mapped.append((k.replace(".bias", ".conv.bias"), vb))
        elif k.startswith("decoder.norm_out."):
            mapped.append((k.replace("decoder.norm_out.", "decoder.conv_norm_out.norm."), vb))
        elif k.startswith("decoder.mid.block_"):
            idx = int(k.split(".")[2].split("_")[1]) - 1
            rest = ".".join(k.split(".")[3:])
            mapped.append(
                (
                    f"decoder.mid_block.resnets.{idx}.{rest}",
                    _tr_conv(vb) if rest.endswith(".weight") and vb.ndim == 4 else vb,
                )
            )
        elif k.startswith("decoder.mid.attn_1."):
            sub = k.removeprefix("decoder.mid.attn_1.")
            if sub.startswith("norm."):
                mapped.append((f"decoder.mid_block.attentions.0.group_{sub}", vb))
            elif sub.startswith(("q.", "k.", "v.")):
                qn, pt = sub.split(".")
                val = vb.squeeze(-1).squeeze(-1) if vb.ndim == 4 else vb
                mapped.append((f"decoder.mid_block.attentions.0.to_{qn}.{pt}", val))
            elif sub.startswith("proj_out."):
                pt = sub.split(".")[1]
                val = vb.squeeze(-1).squeeze(-1) if vb.ndim == 4 else vb
                mapped.append((f"decoder.mid_block.attentions.0.to_out.0.{pt}", val))
        elif k.startswith("decoder.up."):
            parts = k.split(".")
            b_idx = 3 - int(parts[2])
            if parts[3] == "block":
                r_idx = int(parts[4])
                rest = ".".join(parts[5:]).replace("nin_shortcut", "conv_shortcut")
                mapped.append(
                    (
                        f"decoder.up_blocks.{b_idx}.resnets.{r_idx}.{rest}",
                        _tr_conv(vb) if rest.endswith(".weight") and vb.ndim == 4 else vb,
                    )
                )
            elif parts[3] == "upsample":
                rest = ".".join(parts[4:])
                mapped.append(
                    (
                        f"decoder.up_blocks.{b_idx}.upsamplers.0.{rest}",
                        _tr_conv(vb) if rest.endswith(".weight") and vb.ndim == 4 else vb,
                    )
                )

    del raw_vae
    vae = VAE()
    if low_memory:
        vae.decode_mode = VAEBlockDecoder.LOW_MEMORY
    vae.update(tree_unflatten(mapped))
    mx.eval(vae.parameters())
    return vae


def decode_mlx_latents_to_uint8(vae: Any, latents_unpacked: Any) -> np.ndarray:
    """Decode unpacked MLX latents into an `(H, W, 3)` uint8 RGB array."""
    import mlx.core as mx
    from mflux.models.common.vae.vae_util import VAEUtil

    decoded = VAEUtil.decode(vae=vae, latent=latents_unpacked, tiling_config=None)
    mx.eval(decoded)
    if decoded.ndim == 5:
        decoded = decoded[0, :, 0, :, :]
    elif decoded.ndim == 4:
        decoded = decoded[0]
    img_mx = mx.transpose(
        mx.clip((decoded.astype(mx.float32) + 1.0) * 127.5, 0.0, 255.0).astype(mx.uint8),
        (1, 2, 0),
    )
    mx.eval(img_mx)
    return np.array(img_mx)
