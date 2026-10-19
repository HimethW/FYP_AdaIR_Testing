# Log

Every step of the rebuild, in order. Newest at the bottom.
Numbers and comparisons go in [results.md](results.md); this file says what was done and why.

Repo layout: `net/model.py` and the other upstream files stay as released by the AdaIR authors
(needed for benchmarking against the original). All our changes are runtime patches in `patches/`.

---

## 2026-10-19

### Context: old server lost
The server the project ran on went down and all code, checkpoints and data were lost.
Work restarts on **qbits** (RTX 4080 SUPER 16 GB, driver 580.173.02, CUDA 13.0) in `/home/minura/Himeth`.
The feasibility study presentation (101 slides) is the main record of where things stood; its results
are copied into results.md section B.

### Cloned upstream AdaIR
- `git clone https://github.com/c-yn/AdaIR.git` at commit `ccb8b98` ("Update README").
- AdaIR = 28.78M params (checked with a forward pass).

### Created conda env `adair`
- Upstream `env.yaml` pins PyTorch 1.13 + CUDA 11.6 (and a `cpuonly` package), which has no proper
  sm_89 (Ada / RTX 40xx) support, so it was not used.
- New env: Python 3.10, torch 2.4.1+cu121, lightning 2.2.5. Pins and reasons in `requirements-fyp.txt`.
  - numpy 1.23.5: scikit-video 1.1.11 (used for NIQE in `utils/val_utils.py`) needs `np.float`.
  - scipy < 1.14: `utils/imresize.py` imports deprecated `scipy.ndimage.filters`.
  - setuptools 69.5.1: lightning 2.2 imports `pkg_resources`, which newer setuptools removed.
- Verified: CUDA visible (capability 8.9), all repo imports work, AdaIR forward pass on GPU.
- Existing envs on the server (`fyp`, `agrivision`, `en4554`) were not touched.

### Bug: FreModule mask is always empty
In `FreModule.fft` (`net/model.py:337-349`) the low-frequency mask half-size is
`(h // 128 * sigmoid(...)).int()`. The three AFLBs (`fre1/2/3`) run at H/8, H/4, H/2, so
`h // 128` is 0 for every training crop and 0-5 even at 720p, and `int(k * sigmoid)` rounds down.
Measured: mask area 0% at 128, 256, 320x480 and 0.003% at 720x1280 (results.md A1).
Consequences:
- the low-frequency branch is all zeros, the high branch is the full spectrum;
- `.int()` + slice assignment has no gradient, so `rate_conv` gets `grad = None` and never trains.

### Fix: `patches/mask_fix.py` (soft, resolution-independent mask)
- Cutoff = `sigmoid(rate_conv(GAP(x)))`, now read as a fraction of Nyquist per axis (0 = DC, 1 = Nyquist)
  on normalised centred frequency coordinates, so it means the same on 16 px and 640 px maps.
- Soft rectangle: `mask = sigmoid((c_h - |u|)/tau) * sigmoid((c_w - |v|)/tau)`, differentiable in the cutoff.
- `tau = 0.05`, chosen by sweep (results.md A2): coverage within 0.4% of the cutoff and
  d(coverage)/d(cutoff) in 0.94-1.01 down to 16x16 maps. Smaller tau turns into a staircase on small
  maps; larger tau blurs the edge.
- Applied per model instance: `mask_fix.apply(model)` switches each FreModule to `FreModuleMaskFix`
  (same parameters, only `fft` overridden), `mask_fix.remove(model)` switches back. The original
  class is never changed, so original and fixed models can run side by side. No new parameters,
  so released checkpoints load unchanged.
- Each patched block keeps `last_cutoff` and `last_mask_area` for logging and mask plots.
- Verified (results.md A1, A3): mask now covers ~24-26% at every input size, low branch carries
  content, rate_conv gets gradients, with an empty mask the fixed model matches the original
  exactly (only the mask changed), `remove()` restores the original exactly, state_dict keys unchanged.
- Note: AdaIR initialises `para1 = 0` (`net/model.py:298`) and the block returns
  `out * para1 + y * para2`, so in a freshly initialised model the whole frequency branch is off
  until para1 moves. Same as upstream; it is why a gradient check on a fresh model shows zeros.
- Not yet run with the released checkpoint or on real data (not downloaded yet).
  Plan: more teacher changes first, then train from scratch at the end.

### Git repo
- `origin` = `https://github.com/HimethW/FYP_AdaIR_Testing.git` (our work), `upstream` = `c-yn/AdaIR`.
- Push over HTTPS with a fine-grained GitHub token limited to this repo (Contents: read & write).
  Stored by a repo-local credential helper in `~/.git-credentials-fyp`; nothing else on the server
  was reconfigured. (An SSH deploy key was tried first and removed: no SSH keys wanted.)
- `.gitignore` excludes `data/`, `ckpt/`, `*.ckpt|*.pth|*.pt`, `output/`, `AdaIR_results/`, logs, wandb.
  Datasets and checkpoints go to Google Drive instead (link to be added here).
- Added this log and results.md.
- Commit identity: the first push had commits authored with the Claude account's email
  (Achintha Hathurusingha's), so GitHub attributed them to `achintha-hathurusingha`. Fixed by setting
  the repo-local identity to `Himeth Walgampaya <103624187+HimethW@users.noreply.github.com>`,
  re-authoring the three commits and force-pushing over the first push. All commits from here on use
  this identity.
- Push credentials: the first push was not prompted for a token. A GitHub OAuth login for HimethW
  (scopes repo/workflow, i.e. all repos) was supplied from the user's own device through the
  connection to the server, and saved by the `store` helper. It was deleted from the server and
  replaced with a fine-grained token: HimethW, only `FYP_AdaIR_Testing`, Contents read/write,
  expires 2027-03-30. Checked: dry-run pushes to other HimethW repos are denied.
  To rotate it, regenerate on GitHub and save it with:
  `( umask 077; read -rsp "Paste token: " T && echo && printf 'https://HimethW:%s@github.com/HimethW/FYP_AdaIR_Testing.git\n' "$T" > ~/.git-credentials-fyp )`

### Downloaded released checkpoints and Kodak24
- Released AdaIR checkpoints from the README's Google Drive folder
  (`https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH`) into `ckpt/` with
  `gdown --folder` (gdown 5.2.0 added to requirements-fyp.txt). sha256:
  - `adair3d.ckpt` f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb
  - `adair5d.ckpt` e5b8ea892b68057f7d20265ac92585286362b0512c755fddd46681364d17dcf9
  - `adair-single-denoise.ckpt` 17acf9d598a18b5b86e26718377827a98fd09d075c6547b76c188ce9e3bd3f77
  - `adair-single-derain.ckpt` 7a9558ae5a1e096da0d9633f5d38c34270f8aa9f99f680ef1d4a991f0f312a81
  - `adair-single-dehaze.ckpt` 33195f1362c71ad31f27765b29e510aae7be046ef72aeeee109bccee5ad00881
- Kodak24 (`https://r0k.us/graphics/kodak/kodak/kodimNN.png`, 24 PNGs, 768x512) into
  `data/test/denoise/kodak24/` for quick real-image checks.
- Disk: `/home` is 97% full (110 GB free) at this point; keep an eye on it when datasets come in.

### Checked the mask fix with the released weights: `tools/check_mask_fix.py`
Script checks (results.md A4, figure `figures/mask_check_adair3d.png`):
1. The fixed low/high split matches an independent implementation built on the unshifted FFT grid
   (`torch.fft.fftfreq`, no roll/shift) exactly (max diff 0.0) for even and odd sizes, so the mask
   sits on the right frequencies.
2. On Kodak24 + noise sigma 25 with `adair3d.ckpt`: the fixed mask covers ~25% of the spectrum and the
   low band carries 78-89% of the non-DC energy in every AFLB; original: empty in fre1/fre2 on all 24
   images, a 2x2 box around DC in fre3 on 1/24 (fre3 is 256x384 there, so h//128 = 2).
3. Findings about the released model (it was trained with a dead low-frequency path):
   - `rate_conv` weights are still at their random init (KS test vs PyTorch's default U(-b, b) not
     rejected; conv1, which is trained, is clearly rejected). They never got a gradient.
   - The low-path cross-attention temperature (`channel_cross_h`; the names are swapped in
     FreModule.forward: `channel_cross_h` processes the low band) is exactly 0.46 in every head of every
     AFLB: zero queries give zero gradient, so only AdamW weight decay moved it from 1.0.
   - fre3's whole branch is nearly off: para1 mean |.| = 0.0012, contribution 0.0000 of the feature.
4. Dropping the fix into the released weights changes nothing: Kodak24 sigma 25 PSNR 32.53 both
   (max per-image diff 0.001 dB). The cutoffs come out ~0.5 for every image (untrained rate_conv).
   So the fix only pays off with training, which fits the plan (train from scratch at the end).
