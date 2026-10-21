"""Replay a step that gave a NaN / inf gradient in train_fyp.py, to find the operation that produces it.

train_fyp.py saves runs/<run>/nonfinite_r<rank>_s<step>.pt when a step's gradient is non-finite: that process's
micro-batches and (from process 0) the weights before the update. For each micro-batch this runs forward + backward
from those weights. It reports the loss and whether the gradient is finite, the first module whose forward output is
non-finite, and for a non-finite backward the operation autograd's anomaly mode blames (with its forward traceback).

  python tools/repro_nonfinite.py runs/_nan2/nonfinite_r1_s0000025.pt --weights runs/_nan2/nonfinite_r0_s0000025.pt
"""
import argparse
import math
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fyp.models import build_model  # noqa: E402
from patches.losses import PixelFFTLoss  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--weights", default=None, help="dump holding the weights (process 0's), if not in `dump`")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    state = d["model"] if d.get("model") else torch.load(args.weights, map_location="cpu", weights_only=False)["model"]
    model = build_model(d["variant"]).to(args.device).train()
    model.load_state_dict(state)
    loss_fn = PixelFFTLoss(d["fft_weight"], d["fft_norm"]) if d["loss"] == "l1_fft" else None
    print(f"step {d['step']}, variant {d['variant']}, loss {d['loss']}, {len(d['micro'])} micro-batches")

    first_bad = {}

    def hook(mod, inp, out):
        outs = out if isinstance(out, (tuple, list)) else (out,)
        if "name" not in first_bad and any(torch.is_tensor(o) and not torch.isfinite(o).all() for o in outs):
            first_bad["name"] = names[mod]
    names = {m: n for n, m in model.named_modules()}
    for m in model.modules():
        m.register_forward_hook(hook)

    for mb in d["micro"]:
        model.zero_grad(set_to_none=True)
        first_bad.clear()
        lq, gt = mb["lq"].to(args.device), mb["gt"].to(args.device)
        tasks = {t: mb["task"].count(t) for t in mb["task"]}
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                out = model(lq)
                loss = loss_fn(out.float(), gt)[0] if loss_fn else torch.nn.functional.l1_loss(out.float(), gt)
                loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            gnorm = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))))
            print(f"micro {mb['i']}: {tasks} loss {float(loss):.4g} (logged {mb['loss']:.4g}) grad norm {gnorm:.4g}"
                  + ("" if math.isfinite(gnorm) else "  <-- NON-FINITE")
                  + (f"; first non-finite forward output: {first_bad['name']}" if first_bad else ""))
        except RuntimeError as e:
            print(f"micro {mb['i']}: {tasks} -> anomaly in backward"
                  + (f" (first non-finite forward output: {first_bad['name']})" if first_bad else ""))
            print("  " + str(e).splitlines()[0])
            traceback.print_exc(limit=1)


if __name__ == "__main__":
    main()
