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
- 2026-10-20 12:44: training paused at the user's request (a friend needed the GPU): a detached stopper waited for the
  step-8,000 checkpoint, then sent SIGTERM; ours_ft finished its step and saved `last.pt` at step 8,004
  (1.6 epochs), exit code 3, the launcher stopped, GPU free. Resume (when the user says): 
  `setsid nohup bash scripts/run_ours.sh >> runs/ours.out 2>&1 &` (stage 1 returns at once, stage 2 continues from
  step 8,004; remaining ~61 h). The 5-epoch mask-shape watcher keeps waiting.
- 2026-10-20 15:45: resumed after the GPU was free (samitha's jobs had finished). The first resume **crashed** on its
  first step: `train_fyp.py` loaded the checkpoint's EMA weights onto the CPU and the first EMA update mixed CPU and
  GPU tensors ("Expected all tensors to be on the same device"). My resume test before launch had no EMA, so it
  missed this. Nothing was saved after the crash (last.pt still step 8,004). Fix: move the EMA state to the device
  on resume; tested (3 -> 6 steps with EMA across a restart); relaunched 15:48, past step 8,100 at 13.2 img/s,
  ~62 h left (finish ~2026-10-23 05:40). The 5-epoch mask-shape watcher is still waiting (now ~03:10 on 10-21).
- 2026-10-21 03:00: usage check at 4.8 epochs (results.md A16): switching the new frequency branches off changes PSNR by
  only -0.02..+0.03 dB on 6 sets, although they change the decoder features by 2-9%. The new blocks are barely used so
  far. Reported to the user; training continues unchanged for now.
- 2026-10-21 03:30: the 5-epoch watcher produced results/mask_shapes (results.md A17): fre1 shapes pushed to their
  bounds (star angle, aspect, lobe vertex / curvature), fre2 / fre3 near the start; per-degradation differences
  mostly in the dot size, star / lobe nearly the same across degradations.
- 2026-10-21 17:40: the user gave a new folder for Elgin: `/mnt/minura/minura/Himeth`, to be used for everything on Elgin
  from now on. Read-only check: it is empty, owned by minura (rwx), on a local ext4 LVM volume (not NFS), 936 GB in
  total with **413 GB free**. Elgin's root disk, where /home/minura lives, is 99% full (9.4 GB free), so nothing may
  go to the home folder: env, pip/conda/torch/HF caches and TMPDIR all have to live under the new folder. GPUs: 3x
  Quadro RTX 5000 16 GB, all idle. Driver 580.178, CUDA 13.0; system Python 3.12.3; no conda (the old
  /home/minura/miniforge3 is gone). The write and speed tests were not run: Elgin's auto mode blocked writes on the
  mount until the user confirms. With the new folder, disk space no longer rules Elgin out (the 2026-10-20 decision
  rejected it for its 12 GB disk); nothing has been set up there yet.
- 2026-10-21 17:55: with the user's OK ("writing inside /mnt/minura/minura/Himeth on Elgin is allowed"), the Elgin
  write and speed tests ran inside that folder: writing works (files owned by minura); 2 GiB direct I/O writes at
  1.5 GB/s and reads at 2.0 GB/s; 2,000 x 200 KB files in 9.5 s, which was mostly the cost of generating the random
  test data, so many small PNGs are not a problem. All test files were removed and the folder is empty again (413 GB free).

### Baseline moved to Elgin (2026-10-21)
User: "start the baseline [training] on elgin in that folder /mnt/minura/minura/Himeth. keep in mind no fixes just the
raw version we got from the original repo. train on the dataset for 20 epochs as we discussed". This replaces the
2026-10-20 plan (baseline on qbits after ours); ours keeps running on qbits.
- Recipe unchanged from the DECISION: `--variant original` (net/model.py as released, no patch, not even the mask fix;
  checked: no patched module in the model), full released `adair3d.ckpt` (all 28.78M params load, 0 missing / 0
  unexpected), plain L1, aio3_mix, 99,730 iters x 32 crops = 20.0 epochs, lr 1e-4, wd 0.01, 2% warmup, cosine to 1e-6,
  EMA 0.999, true fp32 (Turing has no TF32 anyway), seed 0, same validation / checkpoint cadence as ours.
  `scripts/run_baseline.sh`, run name `baseline_ft`, GPU 0 (GPU 2 carries another user's desktop).
- Getting the data there: a temporary download link on qbits for Elgin to pull from was blocked by auto mode (it would
  expose a local service), so Elgin downloads everything from the original sources with the same scripts
  (`scripts/setup_elgin.sh`: Miniforge + env in the Himeth folder, prepare_data.py, build_mixtures.py, released
  checkpoints with sha256 check). `tools/data_digest.sh` fingerprints every dataset folder (file count, bytes, sha256
  over the md5 of every file); qbits' fingerprint is in `splits/data_digest_qbits.txt`, and setup only passes if
  Elgin's is identical, so both models see exactly the same files.
- `env_elgin.sh` (copied to /mnt/minura/minura/Himeth/env.sh) also points HOME into the Himeth folder: Elgin's `/`
  is 99% full and shared, so no tool may write into /home/minura.
- Speed: one Quadro RTX 5000 was estimated at 2.5-4x slower than qbits for fp32 (to be measured once set up).
- 2026-10-21 17:50: Elgin env ready (python 3.10.22, torch 2.4.1+cu121, 3x Quadro RTX 5000 visible, nothing written
  to /home/minura); setup moved on to the dataset downloads. Note: Elgin's `/` is down to 8.0 GB free (not us; all
  our files are under /mnt/minura/minura/Himeth). Added optional multi-GPU training to `train_fyp.py` (torchrun / DDP,
  restoration stage only; --batch / --accum per GPU, so 2 GPUs x 4 x 4 = the same 32 crops per step; only process 0
  logs, validates and saves; a stop signal on any GPU stops all at the same step; checkpoints interchangeable with
  1 GPU; the single-GPU code path is unchanged). `scripts/run_baseline.sh` takes `GPUS=0,1`. To be tested on Elgin
  (1 vs 2 GPU speed, validation, stop / resume across 2 -> 1 GPU) before the real launch.
- 2026-10-21 18:30: Elgin setup done except the last check: the data fingerprint matched qbits on every folder but
  test/denoise (168 vs 192 files). Kodak24 had been downloaded by hand on qbits (see "Downloaded released checkpoints
  and Kodak24") and prepare_data.py had no step for it. Added a `kodak24` step (r0k.us, kodim01-24.png; kodim01's md5
  matches qbits). Training data, mixtures and adair3d.ckpt (sha256) already matched. Elgin re-runs setup to fetch it
  and re-check, then runs the 1 vs 2 GPU tests.
- 2026-10-21 18:45: Elgin data now identical to qbits on every folder (setup done). Benchmarks of the baseline
  recipe on Elgin:
  - 1 GPU (Quadro RTX 5000, batch 4 x accum 8): ~4.2 img/s (3.3-4.2), 13.0 GB. That is ~10.5 h per epoch, **~8.8
    days for 20 epochs** on one GPU (qbits runs ours at 13.3 img/s). Validation after 60 steps from the released
    weights (EMA, 10 images): bsd68_s25 31.437, rain100l 38.395, sots_unseen 29.959 (released on qbits: 31.442 /
    38.394 / 29.970), haze_noise25 16.02, rain_noise25 24.62, haze_rain 21.10, cdd11 19.36. The pipeline works and
    matches qbits.
  - 2 GPUs: crashed at step 2. DDP found parameters without gradients: in the released AdaIR, 15 FreModule tensors
    never get a gradient (per AFLB: `rate_conv` x2, cut off by the `int()` cast of the mask cutoff, and `conv` /
    `score_gen`, defined but never called). So the original model's mask-size predictor is never trained at all,
    which is another face of the mask bug. Fix: `find_unused_parameters=True`; those grads stay None, so AdamW
    skips them exactly as on one GPU. Re-test running.
- 2026-10-21 19:20: NaN diagnosis on Elgin (NaN detector in train_fyp.py: `--on_nonfinite stop|skip`, logs the
  tasks of each micro-batch and the non-finite tensors, saves the step's batches + weights for
  `tools/repro_nonfinite.py`):
  - 1 GPU: clean. 60 steps (seed 0, GPU0) + 100 steps (seed 1, GPU0), gradient norm 0.08-0.77 throughout.
    Validation unchanged from the released model. On a free GPU0 the speed is **6.8 img/s**: the 4.2 img/s
    benchmark had been sharing GPU0 with a notebook job. So 1 GPU = ~5.4 days for 20 epochs.
  - 2 GPUs (DDP): in 60 steps, garbage averaged gradients at steps 19 and 21 (norm 1e13 / 2e12, finite, applied,
    no clipping), a NaN gradient at step 30 and inf at 33 / 50 / 55 (all skipped). Every GPU's own micro-batch
    losses looked normal, except one 1.3e30 loss on GPU1 at step 33, after the bad updates. 13.4 img/s (1.95x).
  - Reading: not the data, not the model. Suspects: corrupted GPU-to-GPU transfers (NCCL P2P over PCIe) or a faulty
    GPU1 (the 1-GPU runs only used GPU0). Next: `tools/check_nccl.py` (all-reduce results checked against a local
    sum, P2P on / off), GPU1 alone, 2 GPUs with NCCL_P2P_DISABLE=1.
- 2026-10-21 19:25: Elgin system info: GPU0 and GPU1 are joined by an NVLink bridge (NV1, same NUMA node), so NCCL
  uses NVLink P2P between them; ECC is off on all three cards; no Xid / NVRM / AER lines in the kernel log. Before the
  transfer and GPU1 checks could run, GPU1 got busy with another job on the same account (4.3 GB, 80%, 85 C), so they
  were skipped. Decision: start the baseline on GPU0 alone now (1-GPU path verified clean, ~6.8 img/s on a free GPU =
  ~5.4 days). It can move to GPUs 0+1 from any checkpoint once GPU1 is free and `tools/check_nccl.py` (P2P on / off)
  plus a 2-GPU test come out clean. The NaN detector is on in stop mode: a non-finite gradient stops the run before
  the update, so last.pt keeps good weights.

### LAUNCH (2026-10-21 19:10 IST): baseline on Elgin GPU0 (`scripts/run_baseline.sh`, run `baseline_ft`)
Pure original AdaIR, full released adair3d.ckpt (missing 0 / unexpected 0), plain L1, aio3_mix, 99,730 iters x 32
= 20.00 epochs, lr 1e-4, wd 0.01, 2% warmup, cosine to 1e-6, EMA 0.999, true fp32, seed 0, NaN detector in stop mode.
Code at 5db58b2 on Elgin, in /mnt/minura/minura/Himeth/AdaIR (launcher PID 1043601, python 1043611, Elgin clock is
UTC). Step 100: loss 0.031, gnorm 0.37, **6.8 img/s, 13.1 GB, ETA 130 h -> ~2026-10-27 05:30 IST**; no non-finite
gradient. GPU0 runs at 86 C under load. GPU1 has another job on the same account; GPU2 is someone's desktop.
Resume after any stop (on Elgin): `cd /mnt/minura/minura/Himeth/AdaIR && setsid nohup bash scripts/run_baseline.sh
>> runs/baseline.out 2>&1 &` (add `GPUS=0,1` for 2 GPUs once that is verified). Monitoring is through the Elgin
session; qbits' dashboard does not see this run.
- 2026-10-21 20:20: 10-epoch usage check (results.md A18): the new blocks still barely matter (+0.00 to +0.05 dB,
  -0.22 on CDD-11), para1 still ~0.002-0.009. The user asked whether ours already beats the released paper results:
  on the 10-image validation at step 50k, ours is -0.06 dB on BSD68 s25, +0.48 Rain100L, +0.03 SOTS unseen, +6 to +11
  dB on the mixtures (which the released model never saw). A full-set CPU evaluation of step 50k (`eval_fyp.py
  --device cpu`, new option) is running into runs/eval_ours_ft/. Added `tools/check_branch_usage.py`.
- 2026-10-21 20:15: the user asked for results as fast as possible and is fine with 1 GPU. Added `--cudnn_benchmark`
  (cuDNN autotuning, still fp32). The baseline was paused at step 736 (SIGTERM, saved) for an A/B speed test on
  Elgin; it resumes with the flag if that is >= 3% faster.
- 2026-10-21 20:35: cuDNN autotuning A/B on Elgin: under the 3% threshold, so the baseline resumed from step 736
  without it (no progress lost). Note: a resume always starts a new epoch (`epoch = saved epoch + 1`, fresh shuffle),
  so the log's "ep" counter runs ahead after a mid-epoch stop. That's harmless: the run length is fixed in
  iterations (99,730 x 32 crops = 20.0 epochs of samples) and every epoch is a random draw anyway. Same as ours
  after its 8,004 resume.
- 2026-10-21 20:40: full-set CPU evaluation of step 50k stopped (user: "i dont need an evaluation"); the in-training validation is enough for now.
  A/B numbers: 6.78 img/s without vs 6.85 with --cudnn_benchmark (+1%, +0.2 GB, validation 55 -> 70 s, identical losses / val). Baseline resumed 20:19 IST at step 736 (new PIDs: launcher 1066707, python 1066717); step 800 clean, 6.8 img/s, ETA 130 h (~2026-10-27 06:30 IST).
- 2026-10-22 02:10: user: don't interrupt the runs; note ours vs the baseline at every epoch, and at the end pick the
  epoch where ours beats the baseline most (the baseline may catch up later). So every epoch's weights must survive:
  train_fyp.py keeps full snapshots only every 20k steps and overwrites last.pt every 1k. Added `tools/keep_epochs.py`
  (watches train.log; at each multiple of 5,000 copies model + EMA from last.pt, ~0.25 GB, no optimiser). Running
  detached on qbits for ours (runs/ours_ft/epochs; already missed 5k-15k, 25k-35k, 45k-55k, which leaves
  iter_0020000 / iter_0040000 from before) and on Elgin for the baseline (runs/baseline_ft/epochs, from step 5,000).
  Disk: ~2 GB more on qbits (72 GB free), ~4.5 GB on Elgin (413 GB free), so per epoch, not every 2.
- 2026-10-22 10:10: paused ours at the user's request (a friend needs qbits' GPU): SIGTERM to the training process,
  which finished its step and saved `last.pt` at **step 71,089** (14.2 epochs, EMA included), exit code 3; the
  launcher stopped; GPU free (7 MiB). The epoch watcher (keep_epochs.py) and dashboard keep running on the CPU.
  Remaining: 28,641 steps = ~19 h of training. Resume when the user says (check the GPU is free first):
  `setsid nohup bash scripts/run_ours.sh >> runs/ours.out 2>&1 &` (stage 1 returns at once, stage 2 continues).
  The baseline on Elgin is not affected.
- 2026-10-22 12:02: GPU free again (no processes, 7 MiB); ours resumed from step 71,089 with the same launcher. Step
  71,100 clean (loss 0.1025, gnorm 0.47), 12.7 img/s at the start, ETA ~20 h -> ~2026-10-23 08:00. This resume runs the
  current train_fyp.py: same single-GPU training, plus the NaN detector (stop mode) and the gradient norm in the log.

### FINISHED (2026-10-23 07:18): ours_ft, 20 epochs
99,730 iterations done, exit 0, no non-finite gradient in the whole run. Final validation (EMA, 10 images): 31.426 /
38.885 / 30.194 / 24.648 / 30.426 / 28.156 / 31.328 (mean 30.723). Every validation is in results.md A20.
Weights kept: iter_0020000 / 0040000 / 0060000 / 0080000.pt (full) and runs/ours_ft/epochs/step_00{60..95}000.pt +
step_0099730.pt (model + EMA; the final one copied from last.pt by hand). No weights for epochs 1-3, 5-7, 9-11.
qbits GPU is free (7 MiB). The epoch watcher exited with the run. The baseline on Elgin continues (~Oct 27). Next steps
are the user's call ("wait till we are done tomorrow. then i will tell next steps").
- 2026-10-23 10:15-11:50: full test-set evaluations at the user's request ("the standard val to include in the paper"),
  `eval_fyp.py --sets main` (every image, AdaIR protocol, fp32, EMA): ours epoch 20 and the released checkpoint on
  qbits (28 / 25 min), the baseline at epoch 6 on Elgin (step_0030000.pt, 50 min; training paused 05:27-06:17 UTC from
  step 30,025, resumed automatically, nothing lost). Table in results.md A21: ours (epoch 20) beats the baseline (epoch
  6) on 11 of 13 sets, behind only on haze+noise; both fine-tuned models are below the paper on noise / rain (the
  baseline more so) and above it on SOTS. Ours at epoch 8 (iter_0040000.pt) is being evaluated on qbits, as the
  closest saved epoch to the baseline's 6.
- 2026-10-23 19:45: user asked for a full test-set evaluation of the baseline at epoch 8 (step 40,000, due ~02:10 IST Oct 24), then the table re-done as released / baseline ep8 / ours ep8 / diff / ours ep20 / diff. Set up on Elgin as a detached script (pause training after step_0040000.pt is saved, evaluate on GPU0, resume).
- 2026-10-24 00:40: meeting folder /home/minura/Himeth/results/meeting_10_25 (user: 4 slides, only the most important
  things): 1 the mask bug in the released AdaIR (figures/mask_check_adair3d.png), 2 the full-test-set comparison table
  (released / baseline ep6 / ours ep8 / diff / ours ep20 / diff; `tools/compare_table.py`, md + csv + png), 3 visual
  examples released vs ours on the four mixtures (`tools/visual_examples.py`, CPU; one image per set, the one
  closest to the set's median PSNR gain; outputs reproduce the eval PSNRs exactly), 4 learned mask shapes at epoch 20
  (`tools/plot_mask_shapes.py`, CPU, 20 min). README.md there has the one-sentence change bullets, key messages and
  Q&A caveats. The baseline column switches to epoch 8 once Elgin's evaluation is in. The baseline's epoch-6 summary is
  kept locally as runs/eval_baseline/epoch6_main.json (per-image values stay on Elgin).
- 2026-10-24 01:30: frequency analysis at the user's request (how each degradation's frequencies differ, how the
  learned mask parameters move): `tools/freq_analysis.py`, output /home/minura/Himeth/results/frequency_analysis
  (README with the full write-up). All test images, luma 256 crops, Hann window, degraded - clean power in dB:
  - noise adds energy rising with frequency (+5 / +8 / +13 dB at Nyquist), isotropic
  - haze removes ~3.9 dB uniformly at every frequency
  - rain adds a small oriented band along the horizontal-frequency axis (vertical streaks)
  - mixtures add up; CDD-11 has ~2x the haze of our OTS-based set
  Model side, epoch 20 shape params: the dot tracks the input's spectral spread (rank corr. 0.95). It widens with
  noise and narrows with haze, and CDD-11 gets the widest dot. The star / lobe shapes only respond to haze (fre1 lobes
  +7.8% outward, about -1.7 deg; star smaller / flatter), not to noise, and barely to rain. Mixtures add up (haze+rain
  = haze + rain shifts). All shape shifts are under 9% of the allowed range, consistent with the blocks being lightly
  used (A18).
- 2026-10-24 13:30: baseline epoch-8 full evaluation in (Elgin, 50 min; training resumed 22:02 UTC from step 40,012, at step 46,600 by 07:52 UTC, 5.9 img/s, ETA 79 h). results.md A23 + results/meeting_10_25/2_comparison_table now use the baseline at epoch 8 (exactly matched with ours at epoch 8): ours wins 7 of 9; loses noise s50 (-0.10) and haze+noise (-0.15). The epoch-6 table is kept in meeting_10_25/old_baseline_epoch6/.
- 2026-10-24 14:00: teacher vs student comparison at the user's request (the friend's student, S70-CDD11-W16 on Elgin,
  final last.pth at 90k). Read-only look at the friend's folder (fyp-s68) via the Elgin session. Their PREREG says the
  S70 arms are trained from scratch on CDD-11 only, ground truth, no teacher (not distilled from AdaIR, contrary to the
  user's assumption). Their harness uses the same AdaIR metric protocol; the CDD-11 haze_rain test set is byte-identical
  to ours (md5 checked). On it: ours (epoch 20) 29.55 / 0.9615 vs the student 26.62 / 0.9091 (+2.93 dB); the baseline at
  epoch 8 gets 27.78. The student is far behind on BSD68 / Rain100L / SOTS (never trained on them). Output:
  /home/minura/Himeth/results/teacher_vs_student (`tools/teacher_vs_student.py`); the student's reported numbers are
  kept in runs/eval_student/s70_w16_main.json. W24 can be added after its final evaluation (~19:20 IST).
- 2026-10-24 23:00: W24 student (S70-CDD11-W24, 16.6M) final evaluation in (friend's rows, step 90,000, not diverged): CDD-11 haze_rain 27.69 / 0.9300 (ours 29.55, +1.86 dB), 11-type CDD-11 mean 27.01 (W16 26.36). Added to results/teacher_vs_student (5 models; W16-only version kept in old_w16_only/).
- 2026-10-27 01:30: teacher-selection justification for the paper (user: why AdaIR over the newer models with higher
  PSNR). Researched DATPRL-IR, MIRAGE, BaryIR, MoCE-IR, DSwinIR and VLU-Net (papers, repos, commit dates, issues). Two
  code claims were checked by reading MoCE-IR's `src/net/moce_ir.py`: routing noise is added in eval mode, and the eval
  branch is only correct at batch 1. AdaIR measured here: 28.78M params, 162 GMACs at 256x256 (FFT not counted). Write-up
  with criteria, a per-model rejection table, reasons that do NOT hold, and a draft paper paragraph:
  /home/minura/Himeth/results/teacher_selection/README.md. Missing: the date we selected the teacher (for the
  availability argument).
- 2026-10-27 06:10: baseline step 90,000 (epoch 18) validation in (Elgin log 00:40 UTC, no non-finite steps). Added to
  results.md A19: ours - baseline +0.039 / +0.409 / +0.386 / -0.264 / +0.134 / +1.087 / +0.763. Ours wins 6 of 7;
  the haze+noise gap narrowed from -0.513 (epoch 17) to -0.264.
- 2026-10-27 13:40: baseline step 95,000 (epoch 19) validation in (Elgin log 08:06 UTC, no non-finite steps, ETA 7 h).
  Added to results.md A19: ours - baseline +0.059 / +0.407 / +0.309 / -0.379 / +0.132 / +1.123 / +0.707. Ours wins
  6 of 7; haze+noise -0.379 (ours dipped 24.818 -> 24.678).

### FINISHED (2026-10-27 15:08 UTC / 20:38 IST): baseline_ft on Elgin, 20 epochs
- Finished 99,730 iterations, with 0 non-finite steps over the whole run. Final weights are
  /mnt/minura/minura/Himeth/AdaIR/runs/baseline_ft/last.pt. epochs/ holds all 20 epochs: steps 5,000 to 95,000, plus
  step_0099730.pt (231 MB, copied from last.pt in the keep_epochs format, CPU; last.pt untouched).
- Final 10-image validation (results.md A19), ours - baseline at epoch 20: +0.065 / +0.400 / +0.302 / -0.374 / +0.129 /
  +1.098 / +0.792. Mean of 7: baseline 30.379, ours 30.723 (+0.344). Ours wins 6 of 7 at every matched epoch from 16
  to 20; haze+noise is the only loss.
- No full test-set evaluation has been run on the final baseline (waiting for the user's next steps).
- 2026-10-28 00:10: full test-set evaluation of the final baseline (epoch 20, EMA) at the user's request. Run on Elgin
  GPU0 (2995 s); summary in runs/eval_baseline/epoch20_main.json. results.md A24 and
  results/meeting_10_25/2_comparison_table now show released / baseline 8 / ours 8 / Δ / baseline 20 / ours 20 / Δ.
  `tools/compare_table.py --diff` now takes LABEL:REF, so each Δ is against the baseline at the same epoch. Matched at
  epoch 20, ours wins 8 of 9 in PSNR and 9 of 9 in SSIM:
  - noise +0.10 / +0.10 / +0.10, rain +0.04, SOTS +0.22, rain+noise +0.23, haze+rain +0.46, CDD-11 +0.67
  - the only loss is haze+noise (-0.08 PSNR, but +0.006 SSIM)
  - averages: single +0.11, mixed +0.32
  The previous table (ours 20 vs baseline 8) is kept in meeting_10_25/old_ours20_vs_baseline8/.
- 2026-10-28 00:40: paper methodology (teacher part) outline at the user's request, written from the code and run
  configs: /home/minura/Himeth/results/paper/methodology_teacher_outline.md. It covers sections 3.0-3.12 (the AdaIR
  recap, the mask-collapse finding, shaped bands, SGB with its bounds table, per-band CA + BSA, the loss, the two-stage
  DCPT + fine-tune recipe, data composition, the baseline, evaluation, and the student hand-off), plus the figures and
  tables needed and the open points (method name, attributing the gain across bands / FFT loss / DCPT, which epoch to
  report).
- 2026-10-28 01:10: methodology outline rewritten at the user's request: the changes are framed as improvements on AdaIR
  (no bug / fix wording), and every section now explains the idea in plain words before listing what to write. It adds
  a frequency-background primer, an AdaIR-vs-ours comparison table (3.4), and a block diagram of the new AFLB. Same
  file: /home/minura/Himeth/results/paper/methodology_teacher_outline.md.
- 2026-10-29 20:30: paper reference draft at the user's request: paper/method_claude.tex (git-ignored, stays on
  qbits). It is the team's Method section with a full "Frequency-adaptive teacher" subsection added before "Training
  with dynamic teachers", plus Experiments text (baseline, protocol) and supplementary tables (shape bounds, data,
  training settings). Teacher size measured on CPU: 31.13M params and 173 GMACs at 256x256 (AdaIR 28.78M, 162 GMACs,
  FFT not counted). No LaTeX on qbits, so only a static check was run (braces, environments, refs, $ balance).
- 2026-10-29 21:00: paper/method_claude.tex refined at the user's request: methodology only (Experiments text and supplementary tables removed). Shape ranges, stage settings and the data composition are folded into the text. Static LaTeX check clean. Previous version kept in the scratchpad.
- 2026-10-30 14:10: paper/method_claude.tex shortened at the user's request (their compiled methodology came to ~2.5 pages with the student parts still empty). Teacher subsection 1,527 -> 1,038 words, 8 -> 5 display equations, all 5 bullet lists turned into prose. Cut from the method: Restormer channel/block details, the per-degradation dB list (kept as 3 sentences), the full shape-range list (2 examples kept), teacher params/GMACs, data counts and training hyperparameters (listed in a comment, for Experiments -> Implementation details). Two fact fixes vs the previous draft: the spectrum numbers were measured on the test sets (log 2026-10-24 01:30), not the training data; at init each AFLB returns beta*Y (para2 from the released model, ~0.01-0.03), not Y unchanged. Static LaTeX check clean. paper/methodology.tex (the user's file, a copy of the previous draft) not touched. Previous draft kept in the scratchpad.
- 2026-10-30 14:40: plain-language line-by-line explanation of the teacher text written at the user's request: /home/minura/Himeth/results/paper/teacher_method_explained.md (Part A basics: pixels, channels, U-Net, frequency, FFT/spectrum, masks, sigmoid, attention, training words, dB; Part B every sentence and equation of the teacher subsection; Part C the block as a flow + AdaIR vs ours; Part D symbol table). Checked against patches/shaped_bands.py (coordinates, mask formulas, bounds, SGB, BSA, init). Also removed "real" before "haze+rain pairs of CDD-11" in paper/method_claude.tex (CDD-11's mixtures are synthesized by its authors).
- 2026-11-07: new run planned by the user (not started): our teacher on CDD-11, 6 classes only (low, haze, rain,
  haze+rain, low+haze, low+rain), 20 epochs from the released AdaIR weights, to compare with CDD-11 methods. On disk
  (qbits and Elgin) we only have CDD-11 haze_rain + clear (1,183 train / 200 test each); the other 5 classes still
  need downloading. 6 classes = 7,098 train / 1,200 test pairs; 20 epochs at batch 32 is ~4,400 iterations.
  Published 6-class averages (computed from EvoIR Table 4, PSNR/SSIM): EvoIR 29.17/0.898, OneRestore 28.96/0.893,
  WGWSNet 27.00/0.874, PromptIR 26.34/0.867 (all trained on all 11 classes).
- 2026-11-07: research on EvoIR's training trick (arXiv 2512.05104, code github.com/leonmakise/EvoIR) at the user's
  request. "EOS" searches only two loss weights, L = a*L1 + b*(1 - MS-SSIM), a + b = 1, start 0.8/0.2: every 500
  iterations, 5 candidates x 3 generations (keep best, crossover, mutation). Paper: scored on "a fixed stratified
  subset" of a held-out validation set "to reduce variance". Released code: scored on the last training batch only,
  weights clamped to [0.1, 0.9]. In both, the network is frozen while scoring, so the score is linear in a and the
  search always moves to whichever loss term is currently smaller; it does not test how training with a weight would
  turn out. Paper ablation (N+H+R): FMM +0.18, EOS +0.08, both +0.50 dB. No code changes made; waiting for the go.

### CDD-11 run (6 classes), started 2026-11-07 17:30 on qbits
User: train our teacher on CDD-11 (low, haze, rain, haze+rain, low+haze, low+rain) for 20 epochs on top of the
released weights, DCPT first, then "the best training method", an evaluation after every epoch against EvoIR and
OneRestore, equal images per class. First on Elgin, then (user) "dont use elgin for this ... use qbits".
- Data: the 5 missing classes read from the official HF zips (tools/prepare_data.py, CDD_DIRS). qbits now has all 6
  classes + clear: 1,183 train / 200 test each (~10.5 GB more; /home has 24 GB free). The stopped Elgin download left
  partial folders in /mnt/minura/minura/Himeth/AdaIR/data/CDD-11 (clear, haze_rain complete; low, haze partial).
- Held out: 40 training scenes (splits/cdd11_val_scenes.txt, seed 0), all 6 classes of each, never trained on: used
  for validation, DCPT accuracy and the loss-mix search. The test split is only used for the per-epoch table.
- Balance: preset `cdd6` = 1,143 per class (every training scene once per class per epoch), 6,858 per epoch,
  214 steps of 32.
- Start: released AdaIR **5-task** backbone (ckpt/adair5d.ckpt; same architecture, loads the same way as adair3d),
  because it was trained on low-light images (LOL) and half of the 6 classes are low light. A smoke test from our
  3-task DCPT weights scored 11.5 dB on low-light validation crops.
- Stage 1, DCPT (runs/cdd6_dcpt): backbone frozen, 6-class classifier (the CDD classes), focal loss + clean-in /
  clean-out L1, batch 2 x 16, lr 2e-4, **2,000 iterations** (main run: 5,000 over a 23x bigger set; 2,000 = ~9 passes
  over CDD-6; main run's DCPT accuracy was 0.92 at 1,000 and 0.97 at 5,000).
- Stage 2, fine-tune (runs/cdd6_ft): 20 epochs (4,280 steps), batch 4 x 8, lr 1e-4 -> 1e-6 cosine, warmup 2%,
  wd 0.01, EMA 0.999, loss a*L1 + (1-a)*(1-MS-SSIM) + 0.1*FFT-L1 (3-scale MS-SSIM for 128 crops, checked against
  torchmetrics: 0.98587 vs 0.98573), a starts at 0.8 (EvoIR).
- Loss-mix search (our version of EvoIR's EOS): at the start of epochs 2-20, 3 candidate values of a (current +
  2 children of the last search's best two; first search: 0.8 / 1.0 / 0.5) each train a copy of the model for 20
  steps on the same batches; scored by PSNR on 240 centre 256-px crops of the held-out scenes (classes weigh
  equally); the best copy and its 20 steps are kept. Smoke test (64 samples, 2-step look-ahead) ran clean.
- Per-epoch test: EMA weights on the full CDD-11 test set (6 x 200, 1080 x 720), OneRestore's protocol (full image,
  8-bit output, skimage PSNR / SSIM win 7: their img_file/cal_psnr_ssim.py) -> runs/cdd6_ft/test/epoch_XX.json and
  table.md (tools/cdd_table.py, published EvoIR / OneRestore 6-class means 29.17 / 28.96 dB). TF32 inference for
  speed: on 12 images vs true fp32, mean PSNR -0.0004 dB (max 0.011), SSIM max 0.00017; bf16 was rejected (max 0.12
  dB). Full-resolution forward: 1.17 s/img fp32, 0.78 TF32 on the 4080 SUPER.
- Gradient norms on CDD-11 in the smoke test: 2.6-4.7 (main run 0.08-0.77), as the friend saw; clipping stays off
  like the main run (AdamW), NaN detector on.
- Every epoch's weights (model + EMA) kept in runs/cdd6_ft/epochs/.
- Launch: `setsid nohup bash scripts/run_cdd6.sh >> runs/cdd6.out 2>&1 &` (re-running resumes).
- 2026-11-07 17:38: DCPT at step 100: 7.2 img/s (as the main run's DCPT), accuracy 0.47, ETA 2.3 h -> fine-tune from ~20:00, all 20 epochs + tests ~05:30 Nov 8 (estimate: 9 min training + 3 min search + 16 min test per epoch).
- 2026-11-07 20:00: DCPT finished (2,000 iterations). Held-out degradation accuracy 0.85 (low 1.00, haze 0.80,
  rain 1.00, haze+rain 0.70, low+haze 0.80, low+rain 0.80; 10 images per class).
- 2026-11-07 20:25: **first fine-tune attempt stopped after epoch 1.** Two problems:
  1. Writing test/table.md crashed: the detached shell's locale was ASCII and the table header has "Δ"
     (UnicodeEncodeError). Fixed: table.md / train.log written as UTF-8, and `export PYTHONUTF8=1` in the launcher.
  2. Epoch-1 test scores were very low (mean 17.74 dB): the tested EMA weights were still mostly the starting
     weights. EMA 0.999 after 214 steps keeps 0.999^214 = 81% of the start, and the whole run is only 4,280 steps
     (the main run had ~100,000, where 0.999 is fine). Check on 10 held-out images per class (OneRestore protocol,
     TF32): DCPT start 16.63 dB, EMA at step 214 17.93, raw weights at step 214 **23.37**.
  Fix: `--ema_warmup`, EMA decay min(0.999, (1+t)/(10+t)) at step t (the usual warm-up for short runs: follows the
  weights early on, then averages over about the last t/9 steps; 0.958 at step 214, 0.991 at step 1,000, 0.998 at
  the end). Toy check: after 214 steps toward a constant target, warm-up EMA 1.000, plain 0.999 EMA 0.193.
  The first attempt is kept as runs/cdd6_ft_ema999 (epoch-1 weights + test JSON).
- 2026-11-07 20:43: fine-tune restarted from runs/cdd6_dcpt/last.pt with `--ema_warmup` (everything else the same).
  New estimate: epoch 1 test ~21:10, then every ~28 min, all 20 epochs ~06:00 Nov 8.
- 2026-11-07 23:39: CDD-11 fine-tune at epoch 6 of 20 (on schedule). Full-test mean PSNR by epoch (EMA, OneRestore
  protocol): 23.49, 24.64, 25.14, 25.48, 25.78, 25.88 (EvoIR 29.17, OneRestore 28.96); gains shrinking (+0.10 at
  epoch 6). Search kept a = 1.0, 0.893, 0.808, 0.826, 0.769, 0.783. Gap at epoch 6 by class: low -1.2, rain -1.7,
  low+rain -1.9, low+haze -4.5, haze -4.8, haze+rain -5.7: **the haze classes are most of the gap.**
  Held-out scenes at epoch 6, 256-px centre crops (40 scenes, search) vs whole images (5 scenes, epoch val): haze
  32.3 vs 29.2, haze+rain 28.2 vs 25.1, while low / rain / low+haze / low+rain are within ~1 dB. Not the same images,
  so only a hint: haze may be removed well on small regions but not across the whole 1080 x 720 image (we train on
  128-px crops).
- 2026-11-08 00:10: user's target for this model: **at least 28 dB** 6-class mean (EvoIR 29.17, OneRestore 28.96).
  Epoch 7: 26.07, epoch 8: 26.21. Biggest gaps to EvoIR at epoch 7: haze+rain -5.39, haze -4.56, low+haze -4.18
  (76% of the gap together); low+rain -1.84, rain -1.44, low -1.22.
- 2026-11-08 00:43: **scale check** (tools/cdd_scale_check.py, CPU fp32, epoch-6 EMA, 4 test images per class,
  indices 0 / 66 / 133 / 199 of each class; full image = 31.91 on the first image, the same as the logged test).
  Whole image in vs the same image restored in 256-px tiles (64 px overlap, blended): haze 27.55 -> 29.70 (+2.15),
  haze+rain 23.81 -> 25.69 (+1.88), low+haze 21.04 -> 24.45 (+3.41), **rain 33.73 -> 32.93 (-0.80)**. Centre 256 px,
  whole image in vs only that region in: haze 27.66 -> 34.06, haze+rain 24.33 -> 26.07, low+haze 21.65 -> 25.39,
  rain 33.91 -> 33.12. Removing one global colour / brightness offset per image: haze +0.73, haze+rain +1.73,
  low+haze +1.99, rain +1.11.
  Reading: trained on 128-px crops, the model removes haze well at that scale but not across a whole 1080 x 720
  image (part of the error is a wrong overall level for the whole image); rain removal prefers the whole image.
  This is the known train-on-crops / test-on-full-images mismatch of models with global operations (Restormer's
  channel attention over all pixels); TLC (Chu et al., ECCV 2022) is the published test-time fix.
- 2026-11-08 01:35: **test-time inference check** (user: "do your suggestion"). New fyp/local_infer.py: TLC-style
  local aggregation (Chu et al., ECCV 2022): channel attention (and optionally the whole frequency block) computed
  on overlapping windows of `base` px (base/2, /4, /8 at lower levels), outputs averaged; convolutions still see the
  whole image. Checks: window >= image gives exactly the original output (max diff 0); two windows = attention on
  each half (diff 0); model restored after use. Paused cdd6_ft with SIGTERM at step 2246 (epoch 11 of 20, last.pt
  saved) to free the GPU. Epoch 9 26.280, epoch 10 26.316 (whole image).
- 2026-11-08 01:59: sweep on **held-out training scenes** (20 per class, never the test; tools/cdd_infer_sweep.py,
  epoch-10 EMA, TF32): whole image 26.69; tiles 256 / 384 / 512 (1/4 overlap): **27.84** / 27.79 / 27.55; local
  attention 192 / 256 / 384: 26.85 / 27.20 / 27.42; local + frequency blocks 256 / 384: 27.20 / 27.42 (= attention
  only: the frequency blocks' whole-image statistics do not matter). Chosen by the rule fixed beforehand (best
  held-out mean): 256-px tiles.
- 2026-11-08 02:27: 256-px tiles once on the **full CDD-11 test** (epoch 10): **27.10 / 0.8838** vs whole image
  26.32 / 0.8744 (+0.79): low 25.83 (-0.01), haze 29.61 (+1.68), rain 32.12 (-0.78), haze+rain 27.05 (+2.15),
  low+haze 23.62 (+1.61), low+rain 24.38 (+0.06). Still 0.9 below the 28 dB target -> step 3 as agreed.
  fyp/evaluate.py `Tiled` (batched tiles; same placement / blending as run_model's tiling, max diff 4e-7);
  train_fyp.py `--test_tile N` (per-epoch test in N-px tiles); cdd_table notes the tiles.
- 2026-11-08 02:27: **step 3, runs/cdd6_ft256** (scripts/run_cdd6_256.sh): continue from cdd6_ft/last.pt (step
  2246, raw weights) on **256-px crops** (batch 1 x 32, same 32 crops per step), 5 epochs (1,070 steps), lr 5e-5
  (where the 128-px cosine was) -> 1e-6, warm-up 2%, EMA 0.999 with warm-up, loss mix from a = 0.9058 with the
  look-ahead search, per-epoch test in 256-px tiles. Memory 13.3 GB. cdd6_ft stays paused (resumable).
  Estimate: ~36 min training + ~10 min search + ~20 min test per epoch, epoch 1 ~03:25, epoch 5 ~08:00.
- 2026-11-08 03:33: cdd6_ft256 **epoch 1 (256-px crops): 27.28 / 0.8871** (256-px tiles), vs 27.10 for the 128-px
  epoch 10 it started from: low 25.82, haze 29.50, rain 32.03, haze+rain 27.60 (+0.55), low+haze 24.23 (+0.61),
  low+rain 24.51. Speed 3.2 img/s (10 s per step, 36 min per epoch); the tiled test takes 29 min (batching the
  tiles did not shorten it: twice the pixels of the whole image with the 1/4 overlap). Revised finish: epoch 2
  ~04:50, 3 ~06:05, 4 ~07:25, 5 ~08:40.
- 2026-11-08 03:53: user: make it the full 20 epochs. Stage 2 (cdd6_ft256) paused at step 287 (SIGTERM, saved) and
  resumed with `--epochs 10` (2,140 steps; 10.5 at 128 px + 10 at 256 px = 20.5 epochs). The cosine is recomputed
  for the longer run: lr at step 288 4.84e-5 instead of 4.26e-5. Search state (a = 1.0, last search at step 214)
  restored. Expected: epoch 2 ~04:50, then every ~77 min (36 training + 11 search + 29 tiled test), epoch 10 ~15:00.
- 2026-11-08 04:47: cdd6_ft256 **epoch 2: 27.38 / 0.8873** (+0.10): low 25.84, haze 30.00 (+0.50), rain 32.11,
  haze+rain 27.57, low+haze 24.25, low+rain 24.52; only haze moved. (Epoch 2 includes the pause / resume at step 287
  with the 10-epoch schedule.)
- 2026-11-08 06:00: cdd6_ft256 **epoch 3: 27.38 / 0.8878** (+0.00 PSNR, +0.0005 SSIM): low 25.88, haze 30.33 (+0.33),
  rain 31.41 (-0.70), haze+rain 27.79 (+0.22), low+haze 24.32, low+rain 24.58. Gains on haze / haze+rain cancelled by
  rain. The mix search moved a 1.0 -> 0.9051 at step 428 (start of epoch 3). Still 0.62 below 28 dB, 7 epochs to go.
- 2026-11-08 07:13: cdd6_ft256 **epoch 4: 27.47 / 0.8884** (+0.09, best so far): low 25.86, haze 30.20 (-0.13),
  rain 31.85 (+0.44, partly back from epoch 3's drop), haze+rain 27.90 (+0.11), low+haze 24.42 (+0.10), low+rain
  24.60. Mix search a 0.9051 -> 0.8535 at step 642. 0.53 below 28 dB, 6 epochs to go.
- 2026-11-08 08:26: cdd6_ft256 **epoch 5: 27.46 / 0.8885** (-0.01; epoch 4 stays best by PSNR): low 25.90, haze 30.04
  (-0.16), rain 31.79, haze+rain 27.95 (+0.05), low+haze 24.46 (+0.04), low+rain 24.63. Mix search a 0.8535 -> 0.8687
  at step 856. 0.54 below 28 dB, 5 epochs to go (lr now about halfway down the cosine).
- 2026-11-08 09:39: cdd6_ft256 **epoch 6: 27.52 / 0.8889** (+0.05, best so far): low 25.88, haze 30.19 (+0.15),
  rain 31.83, haze+rain 28.09 (+0.14), low+haze 24.49, low+rain 24.62. Mix search a 0.8687 -> 0.8111 at step 1070.
  0.48 below 28 dB, 4 epochs to go.
- 2026-11-08 10:52: cdd6_ft256 **epoch 7: 27.57 / 0.8891** (+0.06, best so far): low 25.88, haze 30.35 (+0.16),
  rain 32.01 (+0.18), haze+rain 28.07, low+haze 24.50, low+rain 24.62. Mix search a 0.8111 -> 0.8946 at step 1284.
  0.43 below 28 dB, 3 epochs to go.
- 2026-11-08 12:05: cdd6_ft256 **epoch 8: 27.63 / 0.8896** (+0.06, best so far): low 25.90, haze 30.46 (+0.11),
  rain 32.13 (+0.12), haze+rain 28.15 (+0.08), low+haze 24.50, low+rain 24.63. Mix search a 0.8946 -> 0.6394 at
  step 1498. 0.37 below 28 dB, 2 epochs to go.
- 2026-11-08 13:18: cdd6_ft256 **epoch 9: 27.59 / 0.8897** (-0.04 PSNR, +0.0001 SSIM; epoch 8 stays best by PSNR):
  low 25.92, haze 30.46, rain 31.86 (-0.27), haze+rain 28.14, low+haze 24.51, low+rain 24.66. Mix search kept
  a = 0.6394 at step 1712. Last epoch (10) ~14:32.
- 2026-11-08 14:37: cdd6_ft256 **finished** (2,140 steps). Epoch 10: 27.59 / 0.8896 (low 25.92, haze 30.41, rain
  31.87, haze+rain 28.17, low+haze 24.52, low+rain 24.67). **Best: epoch 8, 27.63 / 0.8896** (-1.54 vs EvoIR, -1.33
  vs OneRestore, 0.37 below 28 dB); epochs 8-10 within 0.04 dB (levelled off). Final table in results.md A28.
  GPU free. Next steps wait for the user (bigger crops / test-time flips, the latter only with the user's OK).
- 2026-11-08 15:10: user: do both next steps (bigger crops + test-time flips). qbits has one 16 GB GPU (RTX 4080), so
  they run one after the other.
  - fyp/evaluate.py `SelfEnsemble(net, n=8)`: the 8 flips / 90-degree rotations, each undone on the output, averaged
    (geometric self-ensemble, Lim et al. 2017). Checks: identity net -> max diff 1.2e-7; n=1 == plain (diff 0).
    Tiled(SelfEnsemble(net), 256): 11.3 s per 1080x720 image vs 1.4 s (full test ~3.75 h).
    tools/cdd_infer_sweep.py settings tile256_x8 / tile384_x8.
  - train_fyp.py `--grad_ckpt`: the 44 transformer blocks are recomputed in the backward pass (gradients equal to
    4e-7); a 384-px crop needs 5.3 GB for forward + backward instead of not fitting. scripts/run_cdd6_384.sh: from
    cdd6_ft256/last.pt, 384-px crops, 5 epochs, lr 3e-5 -> 1e-6, mix from a = 0.816 (the 256-px stage's last).
  - 15:13: held-out check (20 scenes per class, cdd6val) of cdd6_ft256 epoch 8: tiles 256 / 384, each plain and x8.
    Rule fixed beforehand: the best held-out mean is the setting used on the test.
- 2026-11-08 16:09: held-out check, cdd6_ft256 epoch 8 (20 scenes per class, PSNR mean / SSIM):
  tiles 256 28.374 / 0.8885, **384 28.732** / 0.8885, 256 x8 28.533 / 0.8908, **384 x8 28.892** / 0.8907.
  The 256-px-trained model prefers bigger tiles (+0.36 at 384), and the flips add +0.16 at both sizes. Because 384
  beat 256, also checked (held-out only) 512 28.814 / 0.8880 (best plain), 640 28.517, whole image 28.066 (640 needs
  --tile_batch 2 on 16 GB). Peak at 512 -> 512 x8 being measured (--tile_batch 3; batching does not change results).
  Then: plain + x8 at the chosen tile size, each once on the full test (x8 ~4 h), before the 384-px training.
  tools/cdd_infer_sweep.py takes any tileN / tileN_xK.
- 2026-11-08 16:44: held-out 512 x8: **28.994** / 0.8903 (low 26.47, haze 35.48, rain 34.08, haze+rain 30.03,
  low+haze 23.88, low+rain 24.03) = best held-out -> chosen. Full CDD-11 test, cdd6_ft256 epoch 8, once each:
  512-px tiles (no flips; the like-for-like number vs EvoIR / OneRestore) and 512-px tiles + x8 flips
  (runs/cdd6_infer/test_ft256_ep08_tile512.json). Expected ~17:15 and ~21:20.
- 2026-11-08 17:03: user: "do the best way". Plan changed to save 4 h: the full test stops after 512-px tiles
  without flips; the x8 run on cdd6_ft256 epoch 8 is skipped (its held-out gain is +0.18 at 512 px), and the 384-px
  training (scripts/run_cdd6_384.sh, per-epoch test in 512-px tiles) starts right after (scratchpad chain script).
  At the end: best epoch -> tile size re-checked on held-out scenes -> one full test, with and without x8.
  `Tiled(batch=None)` now picks the tiles per forward from the tile size (8 / 3 / 2 for 256 / 384 / 512).
- 2026-11-08 17:14: **full CDD-11 test, cdd6_ft256 epoch 8, 512-px tiles, no flips: 28.27 / 0.8882** (above the
  28 dB target): low 26.04, haze 32.31, rain 33.30, haze+rain 28.89, low+haze 24.36, low+rain 24.69; -0.91 vs EvoIR,
  -0.69 vs OneRestore. results.md A29. The chain script's process check failed (/proc cmdline is NUL-separated), so
  the sweep was stopped by hand (PID, after the 512 JSON was written) at 17:16; the chain then started the training.
- 2026-11-08 17:16: **runs/cdd6_ft384** started (384-px crops, 5 epochs, --grad_ckpt): 5.7 GB peak, ~1.1 img/s
  (~29 s per step, ~104 min training per epoch); with search (epochs 2-5) and the 512-tile test ~2.75 h per epoch,
  epoch 1 ~19:30, epoch 5 ~06:45 on Nov 9.
- 2026-11-08 19:34: cdd6_ft384 **epoch 1 (384-px crops): 28.27 / 0.8895** (512-px tiles), vs 28.27 / 0.8882 for the
  256-px epoch 8 it started from (same mean, SSIM +0.0013): low 26.02, haze 32.15 (-0.16), rain 32.68 (-0.62),
  haze+rain 29.22 (+0.33), low+haze 24.74 (+0.38), low+rain 24.83 (+0.14). Training 106 min per epoch (1.1 img/s,
  6.0 GB); the 512-tile test 31 min. Epoch 2 (first with the mix search) ~22:20.
- 2026-11-08 22:06: cdd6_ft384 **epoch 2: 28.51 / 0.8899** (+0.24, best so far): low 26.09, **haze 33.09** (+0.94;
  above EvoIR 32.24 and OneRestore 32.52), rain 33.06 (+0.38), haze+rain 29.31 (+0.09), low+haze 24.72, low+rain
  24.80. -0.45 vs OneRestore, -0.66 vs EvoIR. Epoch 3 ~00:50.
- 2026-11-09 00:38: cdd6_ft384 **epoch 3: 28.50 / 0.8900** (-0.02 PSNR, +0.0001 SSIM; epoch 2 stays best by PSNR):
  low 26.12, haze 32.95 (-0.14), rain 32.93 (-0.13), haze+rain 29.37 (+0.06), low+haze 24.77 (+0.05), low+rain
  24.83. Mix search kept a = 1.0 at steps 214 and 428. Epoch 4 ~03:20, epoch 5 ~06:00.
- 2026-11-09 03:11: cdd6_ft384 **epoch 4: 28.55 / 0.8903** (+0.05, best so far): low 26.14, haze 33.05, rain 33.02,
  haze+rain 29.42, low+haze 24.80, low+rain 24.86. -0.41 vs OneRestore, -0.63 vs EvoIR. Mix search a 1.0 -> 0.844 at
  step 642. Epoch 5 (last) ~05:50.
- 2026-11-09 03:20: scripts/cdd6_final.sh started (waits for cdd6_ft384 to finish): best epoch by the per-epoch
  test -> held-out tiles 384 / 512 / 640 / 720 -> chosen tile once on the full test without (if not 512) and with x8
  flips. Log runs/cdd6_final/chain.log. Selection logic dry-checked on the current files (epoch 4; tile 512 for the
  256-px model's held-out file).
- 2026-11-09 05:45: cdd6_ft384 **epoch 5 (last): 28.57 / 0.8904** (+0.02, best of the 384-px stage): low 26.14,
  haze 33.07, rain 33.12 (+0.10), haze+rain 29.43, low+haze 24.81, low+rain 24.85. -0.39 / -0.0026 vs OneRestore,
  -0.60 / -0.0072 vs EvoIR. Stage total vs the 256-px start: +0.30 dB, +0.0022 SSIM. Training finished 05:45.
  scripts/cdd6_final.sh picked epoch 5 at 05:46 and started the held-out tile check (384 / 512 / 640 / 720).
- 2026-11-09 05:58: held-out tile check for cdd6_ft384 epoch 5 (20 training scenes per class, not the test set):
  384 px 28.64 / 0.8891, **512 px 29.07 / 0.8898** (best), 640 px 28.91 / 0.8888, 720 px 28.81 / 0.8884 -> tile 512
  stays (the per-epoch test already used it, so epoch 5 without flips = 28.57 / 0.8904). Full test with the x8 flips
  (tile512_x8) started 05:58, ~4-4.5 h -> runs/cdd6_final/test_ep5_tile512.json.
- 2026-11-09 10:05: **final CDD-11 test, cdd6_ft384 epoch 5, tiles 512 + x8 flips: 28.75 / 0.8927** (4.1 h): low
  26.17, haze 33.35, rain 33.35, haze+rain 29.75, low+haze 24.90, low+rain 24.95. -0.21 / -0.0003 vs OneRestore
  (SSIM level), -0.42 / -0.0053 vs EvoIR. Without flips (like-for-like) 28.57 / 0.8904. results.md A30. qbits GPU free.
- 2026-11-09: ARCHITECTURE.md added: plain-language documentation of our model (backbone, the 4 shaped bands, shape
  generator, band attention, size), training (DCPT, losses, loss-mix search, both experiments' schedules, CDD-11
  progressive crops), testing (protocols, tiles, x8 self-ensemble) and results (all-in-one vs baseline, CDD-11 final),
  checked against the code and results.md.
- 2026-11-09: TEACHER_QA.md added: answers to the paper questions on the teacher runs (CDD-11 class list = the 6
  classes, low+haze+rain also left out; Model A all-in-one and Model B CDD-11 are separate models; data per epoch,
  recipe vs the write-up, DCPT classes, GPU time, CDD-11 / AIO-3 / mixture tables, PSNR / SSIM protocols, what
  ablations exist, teacher feature levels for the student, real learned shape parameters at epoch 20 from
  results/meeting_10_25/mask_shapes_epoch20/shape_params.csv, figure files, checkpoint and eval commands). Numbers
  taken from the saved JSON / CSV files; nothing new was run.
