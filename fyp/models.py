"""Build AdaIR variants and load weights from any of our checkpoint formats.

Variants:
  original          released AdaIR exactly as in net/model.py (no patch at all; with amp_safe=True its
                    frequency step is kept in fp32 so it can run under bf16 autocast, numerically a no-op in fp32)
  maskfix           + soft, resolution-independent rectangle mask (patches/mask_fix.py)
  shaped            + 4 shaped bands, SGB, per-band CA, BSA, bounded shapes (patches/shaped_bands.py)
  shaped_unbounded  same, unbounded shape params (the earlier version, for ablations)
"""
import torch

from net.model import AdaIR
from patches import amp, mask_fix, shaped_bands

VARIANTS = ("original", "maskfix", "shaped", "shaped_unbounded")


def build_model(variant, amp_safe=False):
    model = AdaIR(decoder=True)
    if variant == "original":
        return amp.fp32_frequency(model) if amp_safe else model
    if variant == "maskfix":
        model = mask_fix.apply(model)
        return amp.fp32_frequency(model) if amp_safe else model
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


AFLB_PREFIXES = ("fre1.", "fre2.", "fre3.")


def backbone_only(sd):
    """Drop every AFLB weight except para2. para2 scales the decoder features the AFLB passes on
    (out * para1 + y * para2); the released decoder was trained around its values (|para2| ~ 0.01-0.03),
    so it is part of the backbone's signal path, not of the (broken) frequency branch."""
    return {k: v for k, v in sd.items() if not k.startswith(AFLB_PREFIXES) or k.endswith(".para2")}


def load_weights(model, path, use_ema=False, strict=True, backbone=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = extract_state_dict(ckpt, use_ema)
    if backbone:
        sd = backbone_only(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return ckpt, missing, unexpected


def is_backbone(name):
    """Parameter belongs to the backbone (everything outside the AFLBs, plus para2)."""
    return not name.startswith(AFLB_PREFIXES) or name.endswith(".para2")
