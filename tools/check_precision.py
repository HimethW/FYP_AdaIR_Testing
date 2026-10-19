"""fp32 vs bf16 mixed precision: what is safe and what it buys.

1. Speed and memory of a training step (shaped-band AdaIR + PixelFFTLoss), fp32 vs bf16 autocast,
   at several batch sizes (128 crops).
2. Inference quality: released AdaIR (adair3d.ckpt, frequency step kept in fp32 via
   patches/amp.py) on Kodak24 + noise sigma 25, fp32 vs bf16 autocast: PSNR difference.
3. Numerical drift of one shaped-band forward / backward (bf16 vs fp32), same weights.

Training quality under bf16 can only be judged by a matched training run; this measures the rest.
Usage: python tools/check_precision.py
"""
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.model import AdaIR  # noqa: E402
from patches import amp, shaped_bands  # noqa: E402
from patches.losses import PixelFFTLoss  # noqa: E402

dev = "cuda"


def step_bench(batch, dtype):
    torch.manual_seed(0)
    model = shaped_bands.apply(AdaIR(decoder=True)).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = PixelFFTLoss()
    x, t = torch.rand(batch, 3, 128, 128, device=dev), torch.rand(batch, 3, 128, 128, device=dev)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for i in range(6):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                loss, _ = loss_fn(model(x), t)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if i >= 2:
                times.append(time.time() - t0)
        res = (np.mean(times) * 1000, torch.cuda.max_memory_allocated() / 2 ** 30)
    except torch.OutOfMemoryError:
        res = None
    del model, opt
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def inference_quality():
    model = amp.fp32_frequency(AdaIR(decoder=True))
    sd = torch.load("ckpt/adair3d.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[4:]: v for k, v in sd.items()})
    model = model.to(dev).eval()
    rng = np.random.default_rng(0)
    ps = {"fp32": [], "bf16": []}
    maxdiff = 0
    for p in sorted(glob.glob("data/test/denoise/kodak24/*.png")):
        clean = np.array(Image.open(p).convert("RGB")).astype(np.float64)
        noisy = np.clip(clean + rng.standard_normal(clean.shape) * 25, 0, 255).astype(np.uint8)
        x = torch.from_numpy(noisy).permute(2, 0, 1)[None].float().div(255).to(dev)
        outs = {}
        for name, dt in (("fp32", None), ("bf16", torch.bfloat16)):
            with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                y = model(x).float().clamp(0, 1)
            outs[name] = y
            ps[name].append(peak_signal_noise_ratio(clean / 255, y[0].permute(1, 2, 0).cpu().numpy(), data_range=1))
        maxdiff = max(maxdiff, (outs["fp32"] - outs["bf16"]).abs().max().item())
    d = np.array(ps["bf16"]) - np.array(ps["fp32"])
    return np.mean(ps["fp32"]), np.mean(ps["bf16"]), d.mean(), d.min(), d.max(), maxdiff


def drift():
    torch.manual_seed(0)
    model = shaped_bands.apply(AdaIR(decoder=True)).to(dev).train()
    with torch.no_grad():
        for n in shaped_bands.AFLBS:
            getattr(model, n).para1.fill_(0.1)
    x, t = torch.rand(2, 3, 128, 128, device=dev), torch.rand(2, 3, 128, 128, device=dev)
    loss_fn = PixelFFTLoss()
    out, grads = {}, {}
    for name, dt in (("fp32", None), ("bf16", torch.bfloat16)):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
            y = model(x)
            loss, _ = loss_fn(y, t)
        loss.backward()
        out[name] = y.float().detach()
        grads[name] = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
    rel_out = ((out["bf16"] - out["fp32"]).norm() / out["fp32"].norm()).item()
    cos = torch.nn.functional.cosine_similarity(grads["bf16"], grads["fp32"], dim=0).item()
    rel_g = ((grads["bf16"] - grads["fp32"]).norm() / grads["fp32"].norm()).item()
    return rel_out, cos, rel_g


def main():
    print(f"TF32: cudnn conv {torch.backends.cudnn.allow_tf32}, matmul {torch.backends.cuda.matmul.allow_tf32}")
    print("1) training step, shaped-band AdaIR + pixel/FFT loss, 128x128 crops (ms / step, peak GB)")
    for batch in (2, 4, 6, 8, 12, 16):
        r32 = step_bench(batch, None)
        r16 = step_bench(batch, torch.bfloat16)
        f = lambda r: "OOM" if r is None else f"{r[0]:6.0f} ms {r[1]:5.1f} GB"  # noqa: E731
        speed = f"   bf16 speed-up {r32[0] / r16[0]:.2f}x" if r32 and r16 else ""
        print(f"   batch {batch:>2}:  fp32 {f(r32)}   bf16 {f(r16)}{speed}")
    print("2) released AdaIR inference, Kodak24 sigma 25:")
    p32, p16, dmean, dmin, dmax, maxdiff = inference_quality()
    print(f"   PSNR fp32 {p32:.3f}, bf16 {p16:.3f}; bf16 - fp32 per image mean {dmean:+.3f}, "
          f"min {dmin:+.3f}, max {dmax:+.3f} dB; max pixel diff {maxdiff:.4f}")
    rel_out, cos, rel_g = drift()
    print(f"3) shaped model, one step: output rel. diff {rel_out:.2e}; gradient cosine {cos:.5f}, rel. diff {rel_g:.2e}")


if __name__ == "__main__":
    main()
