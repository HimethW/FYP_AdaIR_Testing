"""Why are the haze classes far behind on the full CDD-11 test? CPU, fp32, EMA weights of one epoch.
  python tools/cdd_scale_check.py runs/cdd6_ft/epochs/epoch_06.pt 4 out.json haze,haze_rain,low_haze,rain
Per image (OneRestore protocol, 8-bit output):
  A full      whole 1080x720 image in, PSNR on the whole image (= the per-epoch test)
  B full@ctr  same output, PSNR on the centre 256x256 region only
  C ctr-only  only the centre 256x256 region in, PSNR on it
  D tiled256  whole image restored in 256-px tiles (64 px overlap, blended), PSNR on the whole image
  Dc          tiled output, PSNR on the centre 256x256 region
  A* / C*     A / C after removing the per-channel mean difference to the clean image (how much of the error
              is one global brightness / colour shift, e.g. a wrong haze level for the whole image)
B vs A: is the centre just an easier region?  C vs B: does the model do worse when it sees a big image?
D/E vs A: would restoring in training-sized pieces help on the real test?"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/minura/Himeth/AdaIR")
from fyp.evaluate import test_sets, load_sample, run_model
from fyp.metrics import psnr_ssim
from fyp.models import build_model

ckpt, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
classes = sys.argv[4].split(",")
torch.set_num_threads(20)
net = build_model("shaped").eval()
net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["ema"])
sets = test_sets("/home/minura/Himeth/AdaIR/data")
q = lambda y: torch.floor(y.clamp(0, 1) * 255 + 0.5) / 255
ten = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255)
S, Y0, X0 = 256, (720 - 256) // 2, (1080 - 256) // 2
res = {}
for c in classes:
    spec = sets[f"cdd11_{c}"]
    idx = np.linspace(0, len(spec["pairs"]) - 1, n).round().astype(int)
    rows = res.setdefault(c, {})
    for i in idx:
        t0 = time.time()
        name, lq, gt = load_sample(spec, int(i), "onerestore")
        x, g = ten(lq), ten(gt)
        with torch.no_grad():
            yA = q(run_model(net, x))
            yC = q(run_model(net, x[..., Y0:Y0 + S, X0:X0 + S]))
            yD = q(run_model(net, x, max_pixels=1, tile=256, overlap=64))
        gc = g[..., Y0:Y0 + S, X0:X0 + S]
        shift = lambda y, t: (y + (t - y).mean(dim=(-2, -1), keepdim=True)).clamp(0, 1)
        r = {"A": psnr_ssim(yA, g)[0], "B": psnr_ssim(yA[..., Y0:Y0 + S, X0:X0 + S], gc)[0],
             "C": psnr_ssim(yC, gc)[0], "D": psnr_ssim(yD, g)[0],
             "Dc": psnr_ssim(yD[..., Y0:Y0 + S, X0:X0 + S], gc)[0],
             "A*": psnr_ssim(shift(yA, g), g)[0], "C*": psnr_ssim(shift(yC, gc), gc)[0]}
        rows[name] = {k: float(v) for k, v in r.items()}
        print(f"{c:10s} {name:12s} " + " ".join(f"{k} {v:6.2f}" for k, v in r.items()) + f"  ({time.time() - t0:.0f}s)",
              flush=True)
        json.dump(res, open(out, "w"), indent=1)
    m = {k: np.mean([v[k] for v in rows.values()]) for k in next(iter(rows.values()))}
    print(f"{c:10s} MEAN         " + " ".join(f"{k} {v:6.2f}" for k, v in m.items()), flush=True)
