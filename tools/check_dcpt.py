"""Functional check of patches/dcpt.py on the shaped-band AdaIR (not a real pre-training run).

Synthetic degradations on Kodak24 crops, 4 classes: noise (sigma 15/25/50), haze (t, A), Gaussian
blur, low light. Images 1-20 train, 21-24 held out.
1. The classification pass stops after decoder_level1 (refinement / output never run).
2. The classification loss alone sends gradient into the encoder and all three AFLBs (incl. SGB).
3. A short DCPT run: held-out classification accuracy before vs after, L_pix and L_cls trend.

Usage: python tools/check_dcpt.py --iters 400 --batch 2
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.model import AdaIR  # noqa: E402
from patches import shaped_bands  # noqa: E402
from patches.dcpt import DCPT  # noqa: E402

dev = "cuda"
CLASSES = ("noise", "haze", "blur", "lowlight")


def load(paths):
    return [torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float().div(255) for p in paths]


def degrade(x, cls, g):
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(1, generator=g).item()  # noqa: E731
    if cls == 0:
        sigma = [15, 25, 50][torch.randint(3, (1,), generator=g).item()]
        return (x + torch.randn(x.shape, generator=g) * sigma / 255).clamp(0, 1)
    if cls == 1:
        t, A = r(0.3, 0.7), r(0.7, 1.0)
        return x * t + A * (1 - t)
    if cls == 2:
        return TF.gaussian_blur(x, kernel_size=15, sigma=r(1.5, 3.0))
    return (x ** r(1.2, 2.0)) * r(0.1, 0.35)


def batch(images, n, size, g):
    lq, gt, y = [], [], []
    for _ in range(n):
        img = images[torch.randint(len(images), (1,), generator=g).item()]
        top = torch.randint(img.shape[1] - size, (1,), generator=g).item()
        left = torch.randint(img.shape[2] - size, (1,), generator=g).item()
        clean = img[:, top:top + size, left:left + size]
        cls = torch.randint(len(CLASSES), (1,), generator=g).item()
        lq.append(degrade(clean, cls, g)), gt.append(clean), y.append(cls)
    return torch.stack(lq).to(dev), torch.stack(gt).to(dev), torch.tensor(y, device=dev)


@torch.no_grad()
def accuracy(dcpt, images, size, g, n=128):
    dcpt.eval()
    correct = per = 0
    per_class = np.zeros((len(CLASSES), 2))
    for _ in range(n // 8):
        lq, _, y = batch(images, 8, size, g)
        pred = dcpt.classify(lq).argmax(1)
        correct += (pred == y).sum().item()
        for c, p in zip(y.tolist(), pred.tolist()):
            per_class[c] += (c == p, 1)
    dcpt.train()
    return correct / n, per_class[:, 0] / np.maximum(per_class[:, 1], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()
    torch.manual_seed(0)
    paths = sorted(glob.glob("data/test/denoise/kodak24/*.png"))
    train_imgs, test_imgs = load(paths[:20]), load(paths[20:])

    net = shaped_bands.apply(AdaIR(decoder=True))
    dcpt = DCPT(net, num_classes=len(CLASSES)).to(dev).train()
    print(f"classifier params {sum(p.numel() for p in dcpt.classifier.parameters()) / 1e6:.2f}M, "
          f"feature dims {[s[0].conv1.in_channels for s in dcpt.classifier.stages]}")

    calls = {"refinement": 0}
    h = net.refinement.register_forward_hook(lambda *a: calls.__setitem__("refinement", calls["refinement"] + 1))
    g = torch.Generator().manual_seed(1)
    lq, gt, y = batch(train_imgs, args.batch, args.size, g)
    with torch.no_grad():
        shapes = [tuple(f.shape[1:]) for f in dcpt.features(lq)]
    print(f"1) features {shapes}; refinement calls during classification: {calls['refinement']}")
    h.remove()

    from patches.dcpt import focal_loss
    with torch.no_grad():
        for n in ("fre1", "fre2", "fre3"):
            getattr(net, n).para1.fill_(0.1)  # as after some training, so the AFLB branch is on
    dcpt.zero_grad()
    focal_loss(dcpt.classify(lq), y).backward()
    reach = {name: getattr(net, name).conv1.weight.grad for name in ("fre1", "fre2", "fre3")}
    print("2) L_cls gradient |sum|: encoder patch_embed {:.2e}, ".format(net.patch_embed.proj.weight.grad.abs().sum().item())
          + ", ".join(f"{k}.conv1 {v.abs().sum().item():.2e}" for k, v in reach.items())
          + f", fre3.sgb.base {net.fre3.sgb.base.grad.abs().sum().item():.2e}")
    dcpt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n in ("fre1", "fre2", "fre3"):
            getattr(net, n).para1.zero_()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    opt_net, opt_cls = dcpt.optimizers()
    ge = torch.Generator().manual_seed(123)
    acc0, per0 = accuracy(dcpt, test_imgs, args.size, ge)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        lq, gt, y = batch(train_imgs, args.batch, args.size, g)
        opt_net.zero_grad(), opt_cls.zero_grad()
        loss, logs = dcpt(lq, gt, y)
        loss.backward()
        opt_net.step(), opt_cls.step()
        hist.append([logs["l_pix"].item(), logs["l_cls"].item(), logs["acc"].item()])
    dt = (time.time() - t0) / args.iters
    ge = torch.Generator().manual_seed(123)
    acc1, per1 = accuracy(dcpt, test_imgs, args.size, ge)
    h = np.array(hist)
    k = max(1, args.iters // 10)
    print(f"3) {args.iters} DCPT iterations, batch {args.batch}, {args.size}px, {dt * 1000:.0f} ms/iter, "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
    print(f"   L_pix first/last {k}: {h[:k, 0].mean():.4f} -> {h[-k:, 0].mean():.4f};  "
          f"L_cls: {h[:k, 1].mean():.3f} -> {h[-k:, 1].mean():.3f};  train acc: {h[:k, 2].mean():.2f} -> {h[-k:, 2].mean():.2f}")
    print(f"   held-out accuracy (4 images, 128 crops, chance 0.25): {acc0:.2f} -> {acc1:.2f}")
    print("   per class before: " + ", ".join(f"{c} {a:.2f}" for c, a in zip(CLASSES, per0)))
    print("   per class after:  " + ", ".join(f"{c} {a:.2f}" for c, a in zip(CLASSES, per1)))
    print(f"   classifier level weights softmax(mix): {[round(v, 3) for v in dcpt.classifier.mix.softmax(0).tolist()]}")


if __name__ == "__main__":
    main()
