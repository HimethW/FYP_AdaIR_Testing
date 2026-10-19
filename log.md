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

### Datasets (`tools/prepare_data.py`, `tools/mixtures.py`, `tools/build_mixtures.py`)
What the old runs used (from the deck): the AdaIR authors' 3-task data (noise: BSD400 + WED -> BSD68;
rain: Rain100L; haze: RESIDE-beta OTS -> SOTS outdoor), mixtures haze+noise sigma 25, rain+noise sigma 25,
haze+rain ("our test") and haze+rain from **CDD-11** (deck slides 88/89/96/98: "Haze + rain (CDD-11 test)",
"v_selfattention (+ mixtures, CDD-11)"), plus O-HAZE / NH-HAZE and a SOTS split into unseen scenes
(424) and "SOTS training copies".

Sources (researched by a subagent; every zip's file list checked over HTTP Range before downloading):

| Data | Source | Size | Notes |
|---|---|---|---|
| BSD400 | INSTALL.md Drive `1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N`, fallback HF mirror `laoduanaaa/Image_Restoration_Datasets` | 29 MB | Drive gave quota errors, mirror used (same size / file list). **Renamed to a<id>.jpg**: data_dir/noisy/denoise.txt lists `a100007.jpg`, the zip has `100007.jpg`; as shipped all 400 BSD images are silently dropped by the loader (AdaIR issue #4) |
| WED | INSTALL.md Drive `1e62XGdi5c6IbvkZ70LFq0KLRhFvih7US`, same HF mirror fallback | 2.2 GB | 4,744 .bmp, names match denoise.txt |
| Rain100L train | INSTALL.md Drive `1tiCYr_-ptzwYlRaQOtsI-7seXkrNo-bs` (RainTrainL.zip) | 119 MB | rain-N / norain-N -> Derain/rainy, Derain/gt (200 pairs) |
| Rain100L test | INSTALL.md Drive `1-3Y7TLZ5EpYYZmFLkRKjV508kcBeibjU` | 53 MB | target renamed norain-NNN -> rain-NNN (test loader needs identical names) |
| Urban100 | INSTALL.md Drive `1UlNulSoyflrEObwu19BBlT7f2_Wxycga` | 135 MB | |
| CBSD68 | github.com/clausmichele/CBSD68-dataset, `CBSD68/original` (sparse checkout) | 5 MB | |
| OTS (RESIDE-beta) | official Box share `25idwrsn890w03grdr6pls28cy38r91i`: part1-4 + clear.zip | 11.9 GB | exactly the 72,135 hazy in hazy_outside.txt + 2,061 clear; clear kept as .jpg (loader reuses the hazy extension) |
| SOTS outdoor | official Box share `uqvnbfo68kns1210z5k5j17cvazavcd1`, SOTS.zip | 435 MB | 500 hazy, 492 gt (<id>.png) |
| O-HAZE / NH-HAZE | ETH NTIRE18 / NTIRE20 official zips | 573 / 331 MB | re-laid out to input/ + target/<NN>.png (O-HAZE GT jpg -> png); usual test splits 41-45 / 51-55 |
| CDD-11 | official HF `gy65896/CDD-11` (OneRestore) | 22.4 + 3.8 GB zips | only `haze_rain` + `clear` read over HTTP Range (1,183 train + 200 test pairs, ~4.2 GB) |

`tools/prepare_data.py` is resumable (each step checks its output), downloads to data/_downloads,
extracts to data/_extract, deletes zips after moving. Bug found and fixed on the first run: the extract
folder for Urban100 was itself called "urban100", so the search returned it and the images were deleted
with the temp folder; `find_dir` now skips the root.

Mixtures (choices; the old synthesis code is lost, so these are documented and reproducible):
- haze + noise: OTS / SOTS hazy + Gaussian noise (AdaIR's convention: clip, uint8). Test: SOTS + sigma 25.
- rain + noise: Rain100L rainy + Gaussian noise. Test: Rain100L test + sigma 25.
- haze + rain ("ours"): the Rain100L streak layer max(rainy - clean, 0) added on top of a real OTS / SOTS
  hazy image (rain is near the camera, in front of the haze), GT = clear. Uses OTS's depth-based haze
  rather than an invented haze. Test: SOTS hazy + Rain100L *test* streaks (no train streaks in the test).
- haze + rain (CDD-11): their pairs.
- Test sets are written once (numpy seed 0, PNG, identical names in input/ and target/); training
  mixtures will be generated on the fly by the training loader with the same functions.
- SOTS / OTS overlap: each SOTS outdoor gt is matched to its most similar OTS clear image (32x32 grey
  cosine, confirmed at 128x128); the seen list goes to `splits/sots_outdoor_seen_in_ots.txt`.
- Download finished on qbits (22 GB in data/). The first big download was killed when the Claude session
  restarted (background jobs die with the session); `prepare_data.py` resumed it (WED / O-HAZE / NH-HAZE had
  finished, CDD-11 continued from file 249, OTS from scratch; one Box connection reset was retried). All
  72,135 OTS hazy images in hazy_outside.txt and their 2,061 clear images present; nothing missing.
  Note: the user's laptop is never used; everything runs and is stored on qbits.
- Checked through AdaIR's own loaders: AdaIRTrainDataset finds all 5,144 denoise images (incl. the 400 BSD),
  200 rain pairs (x120), and the test loaders load every pair of BSD68 / Urban100 / Kodak24 / Rain100L / SOTS /
  O-HAZE / NH-HAZE.
- `tools/build_mixtures.py` built the fixed test sets (data/test/mix, 1.5 GB): haze_noise25 500, rain_noise25 100,
  haze_rain 500, haze_rain_cdd11 200. SOTS / OTS overlap: the best-match cosine is cleanly bimodal (75 scenes at
  >= 0.999, none between 0.98 and 0.999), and the 75 matches are pixel-identical (cosine 1.0, mean abs diff 0.0;
  same IDs as OTS). 76 of the 500 hazy inputs show a scene that is in OTS training, 424 do not: exactly the
  old deck's "SOTS haze (424 valid)". List: `splits/sots_outdoor_seen_in_ots.txt` (tracked in git).

### Training and evaluation code (`fyp/`, `train_fyp.py`, `eval_fyp.py`)
- `fyp/models.py`: `build_model(variant)` for original / maskfix / shaped / shaped_unbounded (original and maskfix
  get the fp32 frequency step so they run under autocast); `load_weights` reads our checkpoints, the released
  Lightning checkpoints and plain state dicts.
- `fyp/data.py`: `TrainSet` with the 7 tasks (denoise, derain, dehaze, haze_noise, rain_noise, haze_rain,
  haze_rain_cdd11), mixtures generated per sample with `tools/mixtures.py`, AdaIR's random_augmentation, DCPT
  labels (6 classes; both haze+rain sources share one). Epoch = fixed count per task (items repeated + random
  remainder, like AdaIR's list repetition). Presets: `aio3` = AdaIR's exact 3-task epoch (111,567 crops),
  `aio3_mix` = aio3 + 12,000 of each mixture (159,567 crops; mixture sizes are our choice).
- `fyp/metrics.py`: PSNR / SSIM like AdaIR's val_utils (RGB, [0,1], skimage) but with `channel_axis`: the
  upstream `compute_psnr_ssim` uses `multichannel=True`, which scikit-image 0.21 no longer has, and crashes
  ("win_size exceeds image extent"), so the upstream test.py cannot compute SSIM in this env.
- `fyp/evaluate.py`: registry of all test sets (bsd68 / urban100 / kodak24 at sigma 15/25/50, rain100l, sots,
  sots_unseen, sots_seen, haze_noise25(+_unseen/_seen), rain_noise25, haze_rain(+_unseen/_seen),
  haze_rain_cdd11, ohaze, ohaze_41_45, nhhaze, nhhaze_51_55; presets paper3 / main / extra). Protocol `adair`
  = centre crop to a multiple of 16 as AdaIR's test.py (default, comparable with the paper); `full` = whole
  image, reflect pad. Denoise noise seeded per image (crc32 of name + sigma), so results don't depend on file
  order. Images over 2.6 MP (O-HAZE) run in 1024 tiles with 64 px linear blending.
- `train_fyp.py`: stages `restore` and `dcpt`; AdamW (AdaIR: lr 2e-4, wd 0.01; DCPT stage: net wd 0,
  classifier lr 5e-5 wd 1e-4), linear warmup (15/180 of the run, AdaIR's ratio) then cosine to 1e-6,
  iteration-based with gradient accumulation (default 4 x 8 = 32, the paper's batch), loss l1 or l1_fft
  (weight, norm), fp32 or bf16, optional EMA and grad clipping, `--init` (e.g. from a DCPT run), automatic
  resume from runs/<name>/last.pt, SIGTERM/SIGINT saves and stops, checkpoints every 1k steps + snapshots every
  20k, validation every 5k on the first 10 images of 7 sets (PSNR) or DCPT top-1 on test images, shape-param
  and band-area logging (to catch mask collapse), TensorBoard + metrics.jsonl + train.log.
  Long runs must be started detached (`setsid nohup ... &`): background jobs die when the Claude session restarts.
- `eval_fyp.py`: evaluate any checkpoint on any sets, JSON with per-image PSNR / SSIM (for paired tests).

Checks (results.md A10-A11): the released adair3d reproduces the paper's Table 1 to <= 0.012 dB, and the old deck's
SOTS unseen / training-copy numbers exactly. Smoke tests passed: restore (shaped, aio3_mix), resume, DCPT stage,
init from DCPT (0 missing keys), bf16, original + EMA, detached run stopped by SIGTERM saves at the current step.

### Precision decision: true fp32 for training and evaluation
Asked by the user (goal = maximum PSNR, not speed). Measured on the shaped model (results.md A12): true fp32 (TF32
off) 74.7 ms / image, TF32 convs (PyTorch's default) 61.1, bf16 autocast 58.8 (41 at batch 8). One conv vs a
float64 reference: fp32 3.2e-7 relative error, bf16 2.9e-3 (~10,000x worse); bf16 inference already cost
0.015 dB on every Kodak image (A9). Effect of bf16 / TF32 on *trained* quality is unknown without an A/B costing
days, so the only zero-risk choice is true fp32: `train_fyp.py` now sets TF32 off by default (`--tf32` to allow
it, `--precision bf16` still available), `eval_fyp.py` always runs true fp32. Re-checked: released AdaIR on
BSD68 sigma 25 / Rain100L gives the same 31.462 / 38.641 as with TF32 (A10). If time gets tight, TF32 is the
first fallback, never bf16 for evaluation.

### DECISION (2026-10-19): how the two models are trained
Agreed with the user. Both start from the released `ckpt/adair3d.ckpt` (150 epochs, 3 tasks; it never saw
mixtures: 15.74 dB on haze+noise) and get the same data and the same training length, so the comparison is fair.
The goal is that our model ends up better than the original.

**Baseline: original AdaIR**, run on the second server (Aberdeen, A5000)
- Model: `--variant original` = net/model.py exactly as released, no patch at all (not even the mask fix).
- Weights: the full released checkpoint, including its own AFLBs.
- Loss: plain L1 (the original recipe). No precautions, no DCPT, no FFT loss.
- Data: `aio3_mix` (AdaIR's 3 tasks + haze+noise, rain+noise, haze+rain, CDD-11 haze+rain). Fine-tune ~20 epochs.

**Ours: shaped-band AdaIR with everything we built**, run on qbits
- Model: `--variant shaped` (mask fix idea -> 4 bounded shaped bands, SGB, per-band CA, BSA).
- Weights: `--init ckpt/adair3d.ckpt --init_backbone`: only the backbone comes from the released model; none of
  its (broken) AFLB weights are used, all three AFLBs start fresh. Exception: `para2`, the decoder feature scale
  (out * para1 + y * para2) that the released decoder was trained around; without it the model drops to ~5 dB
  (results.md A13), so it is backbone, not frequency branch.
- Precautions against the new blocks being ignored (the released model learned to do without its frequency
  blocks): fresh para1 (starts at 0 and opens as it becomes useful); a phase where the backbone is frozen and only
  the new AFLBs train (`--freeze_backbone`). Plan: stage 1 = DCPT with the backbone frozen (trains the new AFLBs +
  the degradation classifier, i.e. the warm-up), stage 2 = restoration fine-tuning of everything.
- Loss: L1 + 0.1 FFT. Data: `aio3_mix`, same ~20 epochs as the baseline in stage 2. True fp32.
- At the end: frequency-branch on/off check, to show the new blocks are actually used.

Both: evaluated with `eval_fyp.py` (presets main + extra; SOTS reported on the 424 unseen images as the main number,
the 76 seen in OTS separately). Exact iteration counts, fine-tuning LR schedule (planned: lr 1e-4, short warmup,
cosine to 1e-6, the same for both) and the DCPT stage length are set at launch. Not started: the user has more
changes to give first.

Code for this: `fyp/models.py` (`original` is now the pure upstream model unless bf16 needs the fp32 frequency
wrapper; `backbone_only`, `is_backbone`), `train_fyp.py` (`--init_backbone`, `--freeze_backbone N|-1`).
Checked (A13): ours at init = released AdaIR with its branch off, to 3 decimals; frozen phase leaves all 497 backbone
tensors bit-identical and trains 5.0M AFLB params; unfreezing trains all 31.13M.

### Everything moved inside /home/minura/Himeth (user rule: only use the Himeth folder on qbits)
Moved / rebuilt inside Himeth and verified (eval reproduces A10 numbers, training with backbone init + freeze runs,
plots work, push works):
- Python env: rebuilt at `/home/minura/Himeth/envs/adair` (same pinned versions, requirements-fyp.txt); the old
  `~/miniforge3/envs/adair` (5.8 GB) removed and its lines taken out of `~/.conda/environments.txt`.
- `/home/minura/Himeth/env.sh` (copy in the repo: `env_qbits.sh`): activates that env and points pip, conda
  packages, torch, CUDA kernel cache, Hugging Face, matplotlib caches and TMPDIR into Himeth. Source it first.
- GitHub token file: `~/.git-credentials-fyp` -> `/home/minura/Himeth/.git-credentials-fyp` (outside the repo
  folder, never committed); repo-local credential.helper updated.
- Removed my files from shared caches: 560 pip cache files (1.94 GB) from today's installs (405 older files kept),
  gdown cookie, matplotlib font cache, 3 torch kernel-cache files.
- Not removed (auto-mode blocked editing the shared conda installation and ~/.ssh; left for the user): 6 conda
  package files from the first env build (python 3.10.22, openssl, ca-certificates, libsqlite, ld_impl,
  _openmp_mutex) in ~/miniforge3/pkgs and their urls.txt lines, and a hashed github.com entry in
  ~/.ssh/known_hosts (added by the first SSH push attempt). ~/.nv/ComputeCache is shared with other GPU jobs on
  this account and can't be separated; from now on CUDA's cache goes to Himeth/.cache/nv.
On Aberdeen only the user's workspaces/Himeth folder may be used; the resource check there only read system info.

### Resources (2026-10-19)
- qbits: RTX 4080 SUPER 16 GB idle, Threadripper 2950X 32 threads, 125 GB RAM, /home 87 GB free (98% full, shared).
  Ours needs ~13.5 GB GPU and ~3 days (DCPT stage + 20 epochs): fits.
- Aberdeen (read-only check via its saved SSH connection): no RTX A5000; 2x Quadro RTX 6000 24 GB + 1x Quadro
  RTX 5000 16 GB, all busy (16.9/24.6, 16.9/24.6, 10.3/16.4 GB used; 100/100/40% util; GPU2 runs another
  AdaIR-distillation job), 40 threads, 93 GB RAM, `/` 93% full, /mnt/sasindi_drive 898 GB free, no conda / unzip,
  Python 3.12, network to GitHub / HF / PyTorch / Box / Drive fine, no scheduler. The baseline (~13 GB) doesn't
  fit next to the current jobs, and an RTX 6000 is ~3x slower than qbits for fp32. Options: find the A5000 host,
  run both on qbits one after the other (~6 days), or get a free GPU on Aberdeen (~8-9 days for the baseline).
- Correction (user): external / mounted drives on the servers (e.g. Aberdeen's /mnt/sasindi_drive) belong to other
  people and must not be used. On Aberdeen only the user's workspaces/Himeth folder, on qbits only /home/minura/Himeth.
- Aberdeen's usable GPU for us is GPU 2, a **Quadro RTX 5000** (Turing, 16 GB; not an A5000). At the check it had
  10.3 GB used by another process (an AdaIR-distillation job), leaving ~6 GB: not enough for the baseline (~13 GB
  at batch 4, fp32) until that job ends. Expected speed vs qbits' RTX 4080 SUPER: fp32 ~11 vs ~52 TFLOPS, memory
  bandwidth 448 vs 736 GB/s -> roughly 2.5-4x slower (estimate, not measured), i.e. ~7-12 days for 20 epochs of
  the baseline vs ~3 days on qbits.
- Elgin (read-only check via its saved SSH connection, 2026-10-20): 3x Quadro RTX 5000 16 GB, all idle (GPU2 carries
  another user's desktop Xorg, avoid it); 40 threads, 93 GB RAM; account minura, /home/minura/Himeth exists (empty);
  but `/` (which holds home) has only **12 GB free (99% full, shared)**; the only other mount is an NFS share that is
  not ours. The baseline needs ~26 GB there (training data 18 GB: OTS 12, WED + BSD400 3.1, CDD-11 train 3.4, rain
  0.1; env 5.8 GB, mostly torch + bundled CUDA libs; released ckpt 0.3; run checkpoints ~1-2 GB). Test sets can stay
  on qbits (evaluate there), and WED as lossless PNG would save ~1.5 GB, but it still doesn't fit, and filling a shared
  disk to 100% would break other people's work and our multi-day run. Same GPU model as Aberdeen's, so ~7-12 days for
  20 epochs (estimate). The baseline on qbits after ours (~day 6) also finishes sooner than either RTX 5000 server.

### DECISION (2026-10-20): both runs on qbits, one after the other
Agreed with the user after checking Aberdeen and Elgin (above): our model first, then the original-AdaIR baseline,
both on qbits' RTX 4080 SUPER (same GPU, same true-fp32 settings, so the comparison is fully matched).
1. Ours: `--variant shaped`, released backbone (+ para2), fresh AFLBs; DCPT stage with the backbone frozen, then
   restoration fine-tuning of everything, L1 + 0.1 FFT, aio3_mix, ~20 epochs. ~3 days.
2. Baseline: `--variant original`, full released checkpoint, L1, aio3_mix, the same ~20 epochs. ~3 days.
Expected: ours done ~day 3, baseline ~day 6 after launch. Both launched detached (`setsid nohup`), auto-resume on
restart. Iteration counts, LR schedule (planned lr 1e-4, short warmup, cosine to 1e-6, same for both) and the DCPT
stage length are fixed at launch. Not started: the user has more changes to give first.

### Training dashboard (`tools/dashboard.py`, `tools/dashboard_template.html`)
Requested by the user, to monitor training. Runs detached on qbits (`setsid nohup python tools/dashboard.py --port 8765`,
pid in /home/minura/Himeth/dashboard/dashboard.pid). Every 15 s it samples the GPU (util, memory, temperature, power,
compute processes + owners), CPU load, RAM and free disk on Himeth's filesystem, reads every runs/<name>/ (args,
metrics.jsonl, train.log, checkpoints, eval*.json) and runs/eval_released/, keeps 7 days of GPU history in
/home/minura/Himeth/dashboard/gpu_history.jsonl, and writes dashboard/data.json + a self-contained index.html.
Served on **127.0.0.1:8765 only** (not reachable from outside); view with an SSH tunnel from the laptop:
`ssh -N -L 8765:127.0.0.1:8765 minura@192.248.10.67`, then http://localhost:8765 (refreshes every 15 s).
Page: GPU / memory / temperature / disk / RAM / CPU tiles (disk warning under 30 GB), run selector, run card (status,
iterations, epochs, ETA, speed, LR, settings, checkpoints), loss / LR / speed charts, DCPT loss / accuracy, in-training
validation PSNR per set, mask health (band areas per AFLB, shape params vs bounds, flags a param at its bound), GPU
history (1h / 6h / 24h / 7d), all-runs comparison table, eval_fyp.py results, log tail. Charts follow the dataviz
guidance (reference palette validated light + dark, legend for >= 2 series, crosshair tooltip, table view, dark mode).
Checked by rendering with headless Chromium (installed in Himeth/.tmp + .cache; libasound unpacked into
Himeth/.tmp/libs, nothing installed system-wide) on two demo runs at 1280 px light/dark and 390 px: no JS errors.
Fixed after looking: faint raw-loss lines were drawn outside their charts (not in the y-range) -> included + clipped;
number formats; epochs < 0.01 shown as 0.00; time-axis ticks snapped to round clock times.

### LAUNCH (2026-10-20): our model, 20 epochs, on qbits (`scripts/run_ours.sh`)
User: "target 20 epochs on both baseline and ours; start the training for our architecture".
Exact settings (both stages: shaped variant, bounded shapes, aio3_mix = 159,567 crops / epoch, 128 crops, true fp32,
TF32 off, AdamW, cosine to the min LR, seed 0):
- Stage 1 `ours_dcpt` (DCPT warm-up): 5,000 iters x (2 x 16) = 160,000 crops = 1.0 epoch; released backbone + para2,
  fresh AFLBs, backbone frozen the whole stage (12.45M trainable: AFLBs + classifier); net lr 2e-4 wd 0, classifier
  lr 5e-5 wd 1e-4, warmup 1/12; focal + L1(clean); DCPT top-1 checked every 500 iters. Measured 7.2 img/s, 8.2 GB
  -> ~6.2 h.
- Stage 2 `ours_ft` (restoration): 99,730 iters x (4 x 8) = 3,191,360 crops = 20.0 epochs; init = stage-1 weights;
  everything trains; L1 + 0.1 FFT (unnormalised, SFNet convention); lr 1e-4, wd 0.01, 2% warmup, cosine to 1e-6;
  EMA 0.999 kept alongside (validation uses EMA; raw weights also saved); validation on 10 images x 7 sets every
  5,000 iters; checkpoints every 1,000 iters, snapshots every 20,000. Measured 13.3 img/s, 13.6 GB -> ~67 h.
- Total ~3 days. The baseline (original, full released weights, L1, same 20 epochs and schedule, EMA too) follows on
  qbits after this finishes.
Robustness fixes made before launch (tested): an interrupted run now exits with code 3 so stage 2 can't start from an
unfinished stage 1; a SIGTERM to the whole process group (e.g. shutdown) used to kill the data-loader workers first and
crash the run unsaved -> now caught wherever it surfaces, the last completed step is saved (or, if it hit mid-update,
the previous periodic checkpoint is kept); data loader closed explicitly at the end. Tested: SIGTERM to the main
process, to the whole group, and to the group mid-step: exit 3 within 4-9 s with the checkpoint saved; normal finish
exit 0. Resume = run `scripts/run_ours.sh` again.

### Results folder and mask shapes at 5 epochs
User: create /home/minura/Himeth/results (outside the repo; holds our results) and, after 5 epochs, a
`results/mask_shapes/` folder showing the mask shapes of each AFLB for each degradation.
- `tools/plot_mask_shapes.py`: runs the first N (20) test images of each degradation (clean, noise s15/25/50,
  rain, haze (SOTS unseen), haze+noise s25, rain+noise s25, haze+rain ours, haze+rain CDD-11; centre 256 crops)
  through a shaped-band checkpoint and writes masks_grid.png (rows fre1/2/3, columns degradations: mean log
  feature spectrum + dot / star / lobe outlines + band areas), masks_<degradation>.png (outlines + filled
  4-band partition per AFLB, area mean / sd), shape_params.csv / .json (mean / sd of every shape param and band
  area), README.md. CPU only (~5.5 s / image with 8 threads, ~18 min for 200 images); training speed unaffected
  (7.1-7.2 img/s during the test).
- `tools/mask_shapes_at_epoch.py`: waits until a run passes N epochs, saves that checkpoint's model + EMA weights
  to <out>/weights_stepXXXXXXX.pt, runs the plot on the EMA weights, exits. Started detached:
  `--run runs/ours_ft --epochs 5 --out /home/minura/Himeth/results/mask_shapes` (log:
  results/mask_shapes_watch.log). 5 epochs of the 20-epoch stage = step 24,932; checkpoints come every 1,000
  steps, so it uses step 25,000 (5.01 epochs), expected ~2026-10-20 23:45.
- Tested on the stage-1 (DCPT) checkpoint at step 1,500 (0.30 epochs), 5 images per degradation: the deepest
  block fre1 already adapts its masks per degradation (dot area 6% clean / 7% rain / 14% haze, haze+noise,
  haze+rain / 20% CDD-11 haze+rain; star slightly rotated for rain), fre2 / fre3 still near the starting shapes.
