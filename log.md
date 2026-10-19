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

### New AFLB: 4 shaped bands + band self-attention (`patches/shaped_bands.py`)
Architecture from the planning diagram ("AdaIR with the modified AFLB*"): FMiM* with 4 shaped masks,
one CA per band, BSA replacing FMoM's H-L / L-H. Read the AdaIR paper (arXiv 2403.14614) first: its mask
is `M_l[H/2 - aH/k : H/2 + aH/k]` with k = 128, so the empty mask is in the paper's own formula; Q of each
band CA comes from the band, K/V from X; the final CA takes Q from X and K/V from X_m. Kept that wiring.

Design (choices not fixed by the diagram are marked *):
- Shapes on normalised centred frequency (0 = DC, 1 = Nyquist), soft edges sigmoid(distance / tau),
  tau = 0.05 (as mask_fix). Same meaning at every AFLB resolution.
- dot: disc, radius = r50 * exp(s0). r50 = radius holding 50% of the non-DC energy of this image's
  feature spectrum, computed per image and detached*; s0 learned. Starts exactly at the 50% radius
  and follows the image (noise pushes it out).
- star: rotated superellipse, params theta, curvature p, size, aspect. Start* p = 0.5 (concave
  4-point star), size 0.75, aspect 1, theta 0.
- lobe: side lobes |v'| >= vertex + curv * u'^2, rotated by theta. Start* vertex 0.6, curv 0.4
  (lobes left/right, arms reach the corners, as in the planning picture).
- rest = everything else. Priority dot > star > lobe*, so M_dot = D, M_star = S(1-D),
  M_lobe = L(1-D)(1-S), M_rest = (1-D)(1-S)(1-L): the 4 masks sum to exactly 1.
- Masks are averaged with their DFT mirror (k -> -k). Without this, rotated shapes on even-size
  maps gave complex bands (imag/real ~0.1) because the Nyquist row/column has no +1 partner.
- SGB: [GAP(P), STD(P)] -> 1x1 (2C -> C/8) -> GELU -> 1x1 (-> 8, zero-init*) -> + learned base
  (initialised to the starting shapes*). Every image starts with the base shapes.
- Shape params decoded with exp (positive ones) / identity (angles), numeric safety clamps only
  (p in [0.1, 10], aspect in [0.01, 100]). Not bounded yet.
- Bands: |IFFT(M_b * F)| (abs, as the original), then one Chanel_Cross_Attention per band
  (q = band, kv = X).
- BSA: + band embedding -> LN (WithBias) -> shared 1x1 Q/K/V -> per pixel, per head (4 heads)
  softmax over the 4 bands -> 1x1 (zero-init*) -> + residual. Starts as identity.
- sum of bands -> 1x1 -> X_m -> CA(q = X, kv = X_m) -> out * para1 + X * para2 (para1 = 0 init,
  as original). FreRefine (H-L / L-H), and the unused `conv` / `score_gen` of the original, dropped.
- `shaped_bands.apply(model)` swaps fre1/2/3 for new untrained FreModuleBands; net/model.py untouched.
- Each AFLB keeps `last_params`, `last_r50`, `last_band_area`, and BSA keeps `last_attn` (B x 4 x 4
  mean mixing) for logging and plots.

Checks (`tools/check_shaped_bands.py`, results.md A5): partition exact (|sum - 1| ~1e-7), bands real
and summing back to P, band areas identical from 32 to 512 px, BSA identity at init, every new parameter
gets gradient, 31.13M params (+2.35M), train step +11%. Figures (`tools/plot_bands.py`):
`figures/bands_start_masks.png` (starting shapes on real feature spectra), `figures/bands_start_kodim.png`
(what each band passes).
Notes:
- Batch 4 at 128x128 already takes 12.5 GB for fwd+bwd on the 16 GB card (original AdaIR too); the
  paper used batch 32. Training here will need mixed precision and/or gradient accumulation.
- The star band picks up the FFT wrap-around edge (bright frame): the image-border discontinuity puts
  energy along the u/v axes, where the star's arms are. Same in the original AdaIR. Possible later
  fix: window or pad before the FFT. Not changed.

### Read the full AdaIR paper (arXiv 2403.14614v1, 28 pages incl. supplementary)
Notes that matter for our changes:
- FMiM: 3x3 conv on the degraded image I (not on the features X; Table 9: image-level 30.52 dB vs
  feature-level 29.29 dB), FFT, MGB = GAP -> 1x1 -> GELU -> 1x1 -> sigmoid -> alpha, beta.
  Low mask `M_l[H/2 - aH/k : H/2 + aH/k, W/2 - bW/k : W/2 + bW/k] = 1`, **k = 128**, "a small value ...
  as the curve junction is relatively small in Fig. 1". So the paper *intends* a tiny low band:
  half-width at most H/128 px = 1/64 (1.6%) of Nyquist. At AdaIR's feature sizes (16-64 px on 128
  crops) that is under one pixel, so the window is empty: the bug is in the paper's formula, not only
  the code. Our mask fix (cutoff = fraction of Nyquist, ~0.5 at init) and the shaped bands use a much
  wider notion of "low"; the dot at the 50%-energy radius (0.03-0.13 of Nyquist on Kodak) is the
  closest analogue of the paper's intended window.
- Band features: Q from the band F_* = IFFT(M_* F), K/V from X (multi-dconv head transposed CA).
- FMoM: H-L = sigmoid(conv7x7([GAP_c, GMP_c](X_h))) gates X_l spatially; L-H = sigmoid(shared
  MLP(GAP_s) + MLP(GMP_s)) of X_l gates X_h per channel; aggregate -> 1x1 -> X_m; final CA with Q from
  X, K/V from X_m. para1 / para2 are in the code only, not in the paper.
- Ablation (Table 7, dehazing, 20 epochs): baseline 28.21; + FMiM with a *fixed* 10x10 square mask
  29.79; + L-H 30.37; + H-L 30.52; full AdaIR (MGB instead of the fixed mask) 31.24.
  Note: with the released code and k = 128, the MGB mask is empty on 128 crops, so the +0.72 dB of
  "MGB over fixed mask" was most likely measured with no low band at all (if the ablation used this
  code). The released weights agree: low path dead (results.md A4).
- Supplementary: AFLBs only in the decoder (encoder + decoder 29.70 vs 30.52); one AFLB per decoder
  level is best (level 2: 28.58, 2+3: 29.83, 2+3+4: 30.52); FMoM vs plain spatial attention
  30.52 vs 29.67 (cross-band interaction matters, which BSA generalises to 4 bands); TB = Restormer
  MDTA + GDFN.
- Training, paper: L1, Adam (0.9, 0.999), lr 2e-4, 150 epochs, batch 32 (all-in-one), 128 crops,
  random flips. Released code: AdamW, LinearWarmupCosineAnnealingLR(warmup 15, max 180) but 150 epochs
  (LR never reaches its minimum), batch 8 per GPU x 4 GPUs = 32. To decide when we write our training.

### Checked the planning diagram against `patches/shaped_bands.py`
Module tree printed from the code (fre2, dim 192, 0.96M params). Every block in the diagram maps to code:

| Diagram | Code | Match |
|---|---|---|
| (a) AFLB* at the 3 decoder positions, fed the degraded image I | `apply()` replaces fre1 / fre2 / fre3; AdaIR.forward calls them with `inp_img` after latent, dec level 3, dec level 2 | yes |
| (b) I -> Conv 3x3 -> FFT -> F | `FreModuleBands.split`: `conv1` (3 -> C), `fft2`, centred | yes |
| (b) P -> SGB (NEW) -> 4 shaped masks | `sgb(p)` -> `decode_params` -> `band_masks` (B, 4, h, w) | yes |
| (b) M_b * F -> IFFT, x4 | `spec * masks`, `ifft2(...).abs()` (abs as in the original) | yes |
| (b) 4 CA, Q = band, K/V = X (red dashed) | `band_ca` = 4 x `Chanel_Cross_Attention`, `ca(band, y)` | yes |
| (c) BSA, per-pixel self-attention over the 4 bands, 4x4, 4 heads | `BandSelfAttention(num_heads=4)`, softmax over bands at every pixel | yes |
| (c) sum -> Conv 1x1 -> X_m | `merge(sum(bands))` | yes |
| (c) CA(Q = X, K/V = X_m) -> X~ | `channel_cross_agg(y, agg)`, then `out * para1 + y * para2` (as original code) | yes |
| (c) H-L and L-H removed | no FreRefine in FreModuleBands | yes |
| (d) CA: 1x1 + Dconv 3x3 for Q/K/V, C x C softmax, 1x1 | unchanged `Chanel_Cross_Attention` | yes |
| (e) SGB: GAP, STD -> concat -> 1x1 -> GELU -> 1x1 -> + learned base -> 8 shape params | `ShapeGenerator`: `[mean, std]` (2C) -> 1x1 (C/8) -> GELU -> 1x1 (8, zero-init) + `base` | yes |
| (f) band emb -> LN -> 1x1 Q/K/V -> softmax over 4 bands -> x V -> Conv 1x1 (zero-init) -> + residual | `band_emb` (4, C) -> LayerNorm WithBias -> `q`/`k`/`v` 1x1 -> einsum -> softmax -> `proj` (zero-init) + input | yes |
| (g) dot: disc, radius at 50% of non-DC energy | `dot_r = r50 * exp(s0)`, r50 per image from the feature spectrum | yes |
| (g) star: rotated superellipse (theta, p, size, aspect) | `star_theta`, `star_p`, `star_size`, `star_aspect` | yes |
| (g) lobe: \|v'\| >= a + c u'^2 (vertex a, curvature c, theta) | `lobe_vertex`, `lobe_curv`, `lobe_theta` | yes |
| (g) rest = 1 - the three | `M_rest = (1-D)(1-S)(1-L)`; masks sum to 1 | yes |

Not specified by the diagram (my choices, listed so they can be changed): starting shapes, priority
dot > star > lobe for the overlaps, tau = 0.05, DFT-mirror symmetrisation, residual taken from the band
inputs before the band embedding, q.k scaled by 1/sqrt(head dim), SGB hidden width C/8, unbounded
shape params with safety clamps.

Commits from here on have no Co-Authored-By trailer (user request).

### Latest-version changes (from the user's final old-server version): bounded shapes, FFT loss, DCPT, precision
The old run with unbounded shape params and a high shape LR collapsed the masks (stars became straight
crossing lines, lobes rotated, the dot vanished in fre2/fre3 and covered 39% in fre1). Four changes:

**1. Bounded mask shapes, normal LR** (`patches/shaped_bands.py`, default `bounded=True`)
- Each raw SGB output is squashed: value = lo + (hi - lo) * sigmoid(raw) (log space for scale-like
  params); the SGB base starts at the logit of the starting shape, so every image still starts at
  exactly the starting shapes. `BOUNDS`:
  dot_scale 0.5-2.0 x r50 (log) | star_theta +-22.5 deg | star_p 0.4-1.0 | star_size 0.4-0.95 |
  star_aspect 2/3-3/2 (log) | lobe_vertex 0.4-0.9 | lobe_curv 0.1-1.5 (log) | lobe_theta +-30 deg.
- star_p first had a lower bound of 0.25; `figures/bands_bounds.png` showed the star degenerating into
  a thin cross (1% area, sub-pixel arms) there, so it was raised to 0.4 (~4-5% area at the limit).
- Shape params train at the normal LR: no LR multiplier anywhere (to be kept so in the training script).
- `bounded=False` keeps the earlier unbounded version for ablations.

**2. Loss: pixel L1 + 0.1 FFT L1** (`patches/losses.py`, `PixelFFTLoss`)
- FFT term exactly as the AdaIR author's SFNet code (checked: difference 0.0): `fft2` over (H, W),
  default unnormalised norm, real/imag stacked, mean L1; weight 0.1. Computed in fp32 under autocast.
- Measured (results.md A7): on 128 crops this FFT term is ~87-91x the pixel term for noise-like errors,
  so it is ~90% of the total loss (95% on 256 crops); ~46% for haze-like errors. With `norm="ortho"` the
  same weight gives ~7% (0.7% for haze). The two are very different objectives; the old run's choice is
  unknown. Default = author's convention; `norm` option added so it can be A/B tested.

**3. DCPT pre-training** (`patches/dcpt.py`)
- Paper: Hu et al., "Universal Image Restoration Pre-training via Degradation Classification", ICLR 2025,
  arXiv 2501.15510; official code github.com/MILab-PKU/dcpt (read in full; it has no license file, so
  this is our own implementation, nothing copied).
- Followed the official code where it differs from the paper text: per iteration one backward on
  L_pix + L_cls, where L_pix = L1(net(gt), gt) (clean image through the whole network) and L_cls = focal
  loss (gamma 2) of a classifier on decoder features of net(lq). Optimisers: net AdamW lr 2e-4 wd 0,
  classifier AdamW lr 5e-5 wd 1e-4. (Paper text: alternating stages, a conv on the deepest feature for
  generation.) Their configs also use EMA 0.999 and a cosine schedule; fine-tune lr 3e-4.
- Features: outputs of AdaIR's decoder_level3 / 2 / 1 (dims 192 / 96 / 96 at H/4, H/2, H), the same hook
  points as their Restormer config (AdaIR shares the backbone). All three AFLBs are upstream, so the
  classification loss also trains the masks / SGB. The classification forward stops after
  decoder_level1 (custom exception in the hook).
- Classifier (7.45M params, only used during pre-training): per level, add softmax-weighted feature,
  2 bottleneck blocks (1x1-3x3-1x1, width 2x, channel LayerNorm), 1x1 to next width + maxpool 2 + ReLU;
  2 more bottlenecks, GAP, linear.
- Labels from AdaIR's de_id: noise levels merged into one class (as their 5D setting); mapping in
  `ADAIR_DE_ID_TO_CLASS`, to be extended for the mixture classes.
- Memory: a DCPT step runs two forward passes, ~12.5 GB at batch 2 (128 crops, fp32).

**4. Precision** (`patches/amp.py`, and FreModuleBands / losses)
- Checked: the released AdaIR fails under bf16 autocast ("Unsupported dtype BFloat16", torch.fft has no
  bf16), and fp16 FFTs fail on non-power-of-2 sizes. So any bf16 use needs the frequency path in fp32,
  which matches what the user remembers from the old server.
- FreModuleBands now always runs its frequency path (conv on I, FFT, masks, IFFT) in fp32; the FFT loss
  too; `amp.fp32_frequency(model)` does the same for the original / mask-fix AFLBs (for baselines).
  This costs no accuracy. Whether the rest of training runs in bf16 is decided by measurement
  (results.md A9) and, for training quality, a matched A/B run; default stays fp32 until then.
- Checks done (results.md A6-A9): bounds hold; FFT term sizes measured; DCPT runs end to end and learns
  (held-out degradation accuracy 0.19 -> 0.82 after 1500 tiny iterations on synthetic degradations);
  bf16 training halves memory (batch 8 fits vs 4) and gives 1.64x images/s, but bf16 *inference* costs
  0.015 dB on every Kodak image with the released weights. Decision for now: evaluation always fp32;
  training precision decided by a matched short A/B run once the datasets are in (fp32 + gradient
  accumulation to batch 32 vs bf16 batch 8 + accumulation).
