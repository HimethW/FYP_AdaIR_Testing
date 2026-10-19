"""Build AdaIR variants and load weights from any of our checkpoint formats.

Variants:
  original          released AdaIR (net/model.py), frequency step kept in fp32 for autocast
  maskfix           + soft, resolution-independent rectangle mask (patches/mask_fix.py)
  shaped            + 4 shaped bands, SGB, per-band CA, BSA, bounded shapes (patches/shaped_bands.py)
  shaped_unbounded  same, unbounded shape params (the earlier version, for ablations)
"""
import torch

from net.model import AdaIR
from patches import amp, mask_fix, shaped_bands

VARIANTS = ("original", "maskfix", "shaped", "shaped_unbounded")


def build_model(variant):
    model = AdaIR(decoder=True)
    if variant == "original":
        return amp.fp32_frequency(model)
    if variant == "maskfix":
        return amp.fp32_frequency(mask_fix.apply(model))
    if variant == "shaped":
        return shaped_bands.apply(model, bounded=True)
    if variant == "shaped_unbounded":
        return shaped_bands.apply(model, bounded=False)
    raise ValueError(f"unknown variant {variant}; choose from {VARIANTS}")


def extract_state_dict(ckpt, use_ema=False):
    """Restoration-network weights from: our train_fyp.py checkpoint, a released AdaIR Lightning
    checkpoint ('state_dict' with 'net.' prefix), or a plain state dict."""
    if isinstance(ckpt, dict) and "model" in ckpt:
        if use_ema:
            if not ckpt.get("ema"):
                raise ValueError("checkpoint has no EMA weights")
            return ckpt["ema"]
        return ckpt["model"]
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("net.") for k in sd):
        sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
    return sd


def load_weights(model, path, use_ema=False, strict=True):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(extract_state_dict(ckpt, use_ema), strict=strict)
    return ckpt, missing, unexpected
