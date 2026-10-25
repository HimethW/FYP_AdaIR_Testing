# Results

All measurements and comparisons. What was done and why is in [log.md](log.md).

- **Section A**: results produced in this rebuild (qbits, from 2026-10-19).
- **Section B**: results from the old server, copied from the feasibility study presentation.
  Code and checkpoints for these are lost and they have **not** been reproduced; kept as targets
  and for comparison. Test-set definitions for some of them (own synthetic haze, mixture sets) are
  not recorded and must be rebuilt before numbers are comparable.

PSNR in dB, higher is better.

---

## A. This rebuild

### A1. FreModule mask coverage, original vs `mask_fix` (2026-10-19)
Random-init AdaIR (no checkpoint), so cutoffs sit near the sigmoid's 0.5. Shows whether the mask exists
at all, not restoration quality. Area = mean of the mask over the spectrum.

| Input | AFLB feature sizes (fre1/2/3) | Original mask area | Fixed mask area (fre1/2/3) |
|---|---|---|---|
| 128x128 | 16 / 32 / 64 | 0% / 0% / 0% | 23.7% / 26.4% / 23.6% |
| 256x256 | 32 / 64 / 128 | 0% / 0% / 0% | 23.7% / 26.4% / 23.5% |
| 320x480 | 40x60 / 80x120 / 160x240 | 0% / 0% / 0% | 23.7% / 26.4% / 23.5% |
| 720x1280 | 90x160 / 180x320 / 360x640 | 0% / 0% / 0.003% | 23.7% / 26.4% / 23.5% |

On a real image (figs/AdaIR.png, 256x256), mean |low branch| original = 0.0000 in all three AFLBs;
fixed = 0.202 / 0.238 / 0.219 (high branch 0.067 / 0.051 / 0.032). Mask is 1.0 at DC, 0.0 at the corners.

### A2. Choice of tau for the soft mask edge (2026-10-19)
1-D coverage vs cutoff over cutoff in [0.15, 0.85]. Ideal: coverage = cutoff, gradient = 1.

| tau | map size | max abs(coverage - cutoff) | d(coverage)/d(cutoff) min / max |
|---|---|---|---|
| 0.01 | 16 | 0.028 | 0.05 / 3.13 |
| 0.01 | 32 | 0.005 | 0.51 / 1.59 |
| 0.02 | 16 | 0.011 | 0.50 / 1.59 |
| 0.03 | 16 | 0.004 | 0.83 / 1.17 |
| **0.05** | **16** | **0.004** | **0.94 / 1.01** |
| 0.05 | 32 / 64 / 640 | 0.003 / 0.003 / 0.002 | 0.95 / 1.00 (all) |
| 0.08 | 16 | 0.013 | 0.85 / 1.00 |
| 0.08 | 640 | 0.011 | 0.87 / 1.00 |

Chosen: **tau = 0.05** (16 px is the smallest map: fre1 on 128 crops).

### A3. `mask_fix` correctness checks (2026-10-19)
para1 set to 0.5 in all AFLBs to mimic a trained model (it is 0 at init, which blocks all gradient to the branch).

| Check | Original | Fixed |
|---|---|---|
| grad reaching rate_conv (fre1 / fre2 / fre3) | None / None / None | 1.5e-07 / 4.9e-07 / 9.0e-07 |
| Fixed model with mask forced to zero vs original, max abs diff | n/a | 0.0 |
| Fixed model with real mask vs original, max abs diff | n/a | 2e-4 (mask has an effect) |
| `remove()` vs original, max abs diff | n/a | 0.0 |
| state_dict keys | n/a | identical |

### A4. Mask fix with the released `adair3d.ckpt` on Kodak24, noise sigma 25 (2026-10-19)
`python tools/check_mask_fix.py --ckpt ckpt/adair3d.ckpt --images data/test/denoise/kodak24 --sigma 25`
(24 images, 768x512, noise as in `DenoiseTestDataset`, numpy seed 0). Figure: `figures/mask_check_adair3d.png`.

Fixed low/high split vs an independent reference on the unshifted FFT grid: max abs diff **0.0** at
16x16, 32x48, 64x64, 33x47 (odd), 127x128.

Per AFLB, mean over the 24 images:

| AFLB (feature size) | Model | Mask area | Images with a mask | Non-DC energy in low band | mean abs low | mean abs high | Cutoff h / w (range) |
|---|---|---|---|---|---|---|---|
| fre1 (64x96) | original | 0.000% | 0/24 | 0.000% | 0.0000 | 0.0931 | hard mask |
| fre1 (64x96) | fixed | 25.07% | 24/24 | 78.0% | 0.0904 | 0.0159 | 0.495 / 0.506 (0.492-0.510) |
| fre2 (128x192) | original | 0.000% | 0/24 | 0.000% | 0.0000 | 0.1390 | hard mask |
| fre2 (128x192) | fixed | 24.45% | 24/24 | 88.6% | 0.1369 | 0.0148 | 0.493 / 0.496 (0.489-0.497) |
| fre3 (256x384) | original | 0.000% | 1/24 | 0.094% | 0.0030 | 0.0834 | hard mask |
| fre3 (256x384) | fixed | 24.82% | 24/24 | 87.3% | 0.0847 | 0.0110 | 0.498 / 0.498 (0.497-0.501) |

How much each AFLB changes the decoder feature, |agg * para1| / |y * para2|:

| | fre1 | fre2 | fre3 |
|---|---|---|---|
| original | 0.0326 | 0.1235 | 0.0000 |
| fixed | 0.0298 | 0.1248 | 0.0000 |
| para1 mean abs (released) | 0.0226 | 0.0380 | 0.0012 |
| para2 mean abs (released) | 0.009 | 0.026 | 0.030 |

Released weights vs PyTorch default init U(-b, b), b = 1/sqrt(fan_in) (KS test):

| Layer | n | max abs w / b | std / (b/sqrt3) | KS p |
|---|---|---|---|---|
| fre1.rate_conv.0 | 18432 | 1.00 | 1.00 | 0.090 |
| fre1.rate_conv.2 | 96 | 1.00 | 0.93 | 0.41 |
| fre2.rate_conv.0 | 4608 | 1.00 | 1.00 | 0.46 |
| fre2.rate_conv.2 | 48 | 0.99 | 0.91 | 0.053 |
| fre3.rate_conv.0 | 1152 | 1.00 | 1.01 | 0.81 |
| fre3.rate_conv.2 | 24 | 0.73 | 0.66 | 0.029 |
| fre1.conv1 (trained, for contrast) | 10368 | 0.77 | 0.49 | 0 |
| fre2.conv1 (trained) | 5184 | 0.90 | 0.53 | 2e-202 |
| fre3.conv1 (trained) | 2592 | 0.46 | 0.46 | 6e-169 |

rate_conv is consistent with never having been trained (fre3.rate_conv.2's p = 0.029 is one of 9 tests on
24 weights, not significant after correction); conv1 is clearly trained.

Cross-attention temperatures in the released model (4 heads each). `channel_cross_h` is the one that
processes the **low** band (names are swapped in FreModule.forward):

| AFLB | channel_cross_l (high band) | channel_cross_h (low band) | channel_cross_agg |
|---|---|---|---|
| fre1 | 0.69, 0.70, 0.66, 0.42 | 0.46, 0.46, 0.46, 0.46 | 0.50, 0.55, 0.50, 0.52 |
| fre2 | 0.71, 0.68, 0.69, 0.70 | 0.46, 0.46, 0.46, 0.46 | 0.54, 0.54, 0.60, 0.52 |
| fre3 | 0.46, 0.46, 0.46, 0.46 | 0.46, 0.46, 0.46, 0.46 | 0.44, 0.41, 0.42, 0.41 |

0.46 everywhere = init 1.0 shrunk by weight decay alone (zero queries give zero gradient): the low path,
and all of fre3, never learned.

Denoising with the released weights, no retraining (Kodak24, sigma 25):

| | PSNR | SSIM |
|---|---|---|
| noisy input | 20.44 | 0.3502 |
| released AdaIR (original mask) | 32.53 | 0.8875 |
| released AdaIR + mask_fix | 32.53 | 0.8875 |

fixed - original per image: mean -0.000 dB, min -0.001, max +0.000. The fix does nothing without retraining:
the cutoffs are untrained (~0.5 for every image) and the rest of the network was trained around an empty mask.

---

## B. Old server results (from the feasibility presentation, not reproduced)

### B1. Teacher ONNX export attempts (slide 27)
| # | Model | Exporter | Opset | Result |
|---|---|---|---|---|
| A0 | unpatched | TorchScript | 17 | rank error @ FreModule |
| A17 | mask patched | TorchScript | 17 | fft_fft2 unsupported |
| A20 | mask patched | TorchScript | 20 | fft_fft2 unsupported |
| B18 | unpatched | dynamo | 18 | export failed |
| B20 | unpatched | dynamo | 20 | export failed |

Causes: dynamic slicing from the learned threshold (graph not static) and complex `torch.fft.fft2` (no ONNX op).

### B2. Mask fix + retrain vs released AdaIR (slides 63-65)
Full held-out sets. Dehazing here is our own synthetic haze, not SOTS. The deraining drop was attributed
to the fine-tuning data, not the fix (the control arm lost the same).

| Task | Released AdaIR | After fix + retrain | Change |
|---|---|---|---|
| Deraining | 38.62 | 38.33 | -0.28 |
| Denoise s15 | 34.10 | 34.13 | +0.02 |
| Denoise s25 | 31.45 | 31.47 | +0.02 |
| Denoise s50 | 28.18 | 28.20 | +0.02 |
| Dehazing | 22.80 | 28.36 | +5.57 |
| **Mean** | 31.03 | **32.10** | **+1.07** |

Learned mask coverage after the fix (AFLB1 / AFLB2 / AFLB3): denoise s25 40.4 / 19.8 / 28.2%,
dehaze 49.2 / 16.9 / 29.5%, derain 40.9 / 19.7 / 28.3%. Released AdaIR: 0.0% everywhere.

Mask on vs mask off, identical weights:

| Task | Mask off | Mask on | Gain | Images improved | p |
|---|---|---|---|---|---|
| Dehazing | 28.118 | 28.362 | +0.244 | 64.7% | 3.3e-03 |
| Deraining | 38.296 | 38.332 | +0.037 | 71.0% | 8e-05 |
| Denoise s15 | 34.128 | 34.128 | -0.000 | 54.4% | 0.81 n.s. |
| Denoise s25 | 31.469 | 31.470 | +0.000 | 54.4% | 0.36 n.s. |
| Denoise s50 | 28.202 | 28.203 | +0.001 | 76.5% | 2e-05 |
| **Overall** | 32.043 | 32.099 | +0.056 | 64.8% (241/372) | 4e-08 |

The mask is genuinely used but the gain is small; almost all of it is dehazing. The AdaIR paper's Table 7
attributes +1.58 dB to this module.

### B3. Teacher ablation steps 1-9, trained on single tasks + mixtures (slide 70)
Held-out test sets. Noise = average of s15/25/50.

| Step | Variant | Haze+noise | Haze | Noise (avg) | Rain | H+N vs original | H+N vs previous |
|---|---|---|---|---|---|---|---|
| 1 | Original AdaIR (released) | 15.74 | 31.05 | 31.24 | 38.62 | - | - |
| 2 | Mask off, trained on the mix | 22.00 | 30.91 | 30.90 | 38.21 | +6.26 | vs 1: +6.26 (p<0.001) |
| 3 | **Mask on (repaired mask)** | **22.38** | 30.93 | 30.98 | 38.21 | +6.64 | vs 2: +0.39 (p=0.004) |
| 4 | L-H gate removed | 19.76 | 30.96 | 30.44 | 38.07 | +4.02 | vs 3: -2.62 (p<0.001) |
| 5 | H-L gate removed | 22.07 | 30.91 | 30.96 | 38.22 | +6.33 | vs 3: -0.32 (p<0.001) |
| 6 | Threshold by energy (cut where share q of energy lies, q starts 50%) | 22.28 | 30.92 | 30.87 | 38.25 | +6.54 | vs 3: -0.11 (p=0.922) |
| 7 | Per-band weights (one weight per band, chosen per photo) | 22.38 | 30.94 | 30.85 | 38.22 | +6.64 | vs 3: -0.00 (p=0.002) |
| 8 | Per-band weights that also see noise (spread) | 22.54 | 30.92 | 30.83 | 38.23 | +6.80 | vs 7: +0.16 (p=0.039) |
| 9 | Energy threshold + noise-aware weights | 22.21 | 30.94 | 30.85 | 38.25 | +6.47 | vs 8: -0.33 (p=0.012) |

### B4. Rectangle mask vs arbitrary (shaped) masks (slide 81)
"Arbitrary mask" = the 4 shaped bands (dot / star / lobe / rest).

PSNR:

| Test | Original AdaIR | Step 3: rectangle mask | Step 8: rectangle + band weights + spread | Arbitrary mask |
|---|---|---|---|---|
| Rain | 38.62 | 38.21 | 38.23 | 38.32 |
| Noise s15 | 34.10 | 33.83 | 33.61 | 33.83 |
| Noise s25 | 31.45 | 31.24 | 31.11 | 31.20 |
| Noise s50 | 28.18 | 27.88 | 27.78 | 27.94 |
| Haze | 31.05 | 30.93 | 30.92 | 30.93 |
| Haze + rain | 20.25 | 20.35 | 20.38 | 20.31 |
| Haze + noise | 15.74 | 22.38 | 22.54 | 22.38 |
| Rain + noise | 24.51 | 23.55 | 23.52 | 23.62 |
| Mean, single tasks | 32.68 | 32.41 | 32.33 | 32.45 |
| Mean, mixtures | 20.17 | 22.09 | 22.15 | 22.10 |

SSIM:

| Test | Original AdaIR | Step 3 | Step 8 | Arbitrary mask |
|---|---|---|---|---|
| Rain | 0.9828 | 0.9815 | 0.9815 | 0.9819 |
| Noise s15 | 0.9347 | 0.9331 | 0.9324 | 0.9331 |
| Noise s25 | 0.8914 | 0.8898 | 0.8895 | 0.8895 |
| Noise s50 | 0.8016 | 0.7991 | 0.7986 | 0.7998 |
| Haze | 0.9796 | 0.9789 | 0.9789 | 0.9791 |
| Haze + rain | 0.8205 | 0.8154 | 0.8155 | 0.8173 |
| Haze + noise | 0.7168 | 0.7508 | 0.7480 | 0.7626 |
| Rain + noise | 0.7371 | 0.7194 | 0.7184 | 0.7206 |

### B5. Mask shape learning rate (slide 82)
| Test | Arbitrary mask | Shape LR x10 | Shape LR x100 |
|---|---|---|---|
| Rain | 38.32 | 38.32 | 38.41 |
| Noise s15 | 33.83 | 33.84 | 33.91 |
| Noise s25 | 31.20 | 31.23 | 31.29 |
| Noise s50 | 27.94 | 27.83 | 28.01 |
| Haze | 30.93 | 30.88 | 30.94 |
| Haze + rain | 20.31 | 20.32 | 20.30 |
| Haze + noise | 22.38 | 22.47 | 22.30 |
| Rain + noise | 23.62 | 23.72 | 23.66 |
| Mean, single tasks | 32.45 | 32.42 | 32.51 |
| Mean, mixtures | 22.10 | 22.17 | 22.09 |

### B6. From-scratch AdaIR variants after epoch 5 of 20 (slide 88)
`*` = v_selfattention differs from v_maskfix per image (p < 0.05).

| Test | AdaIR (single-task data) | AdaIR (+ mixtures) | v_maskfix (+ mixtures) | v_selfattention (+ mixtures, CDD-11) |
|---|---|---|---|---|
| Rain | 33.09 | 33.16 | 33.03 | 33.19* |
| Noise s15 | 33.45 | 33.09 | 33.34 | 33.10* |
| Noise s25 | 30.78 | 30.49 | 30.70 | 30.54* |
| Noise s50 | 27.41 | 27.14 | 27.30 | 27.20* |
| Haze | 28.73 | 28.95 | 28.71 | 27.91* |
| Haze + noise | 15.72 | 24.20 | 21.08 | 22.37* |
| Rain + noise | 24.49 | 28.72 | 28.81 | 28.66* |
| Haze + rain (our test) | 20.87 | 25.86 | 25.84 | 19.98* |
| Haze + rain (CDD-11 test) | 20.40 | 20.31 | 20.18 | 22.02* |
| Mean, single tasks | 30.69 | 30.57 | 30.61 | 30.39 |

### B7. Best student vs v_selfattention vs original AdaIR (slide 96)
v_selfattention at epoch 5 of 20. Brackets: S54 vs original AdaIR.

| Test | v_selfattention | S54 (student) | Original AdaIR |
|---|---|---|---|
| Rain100L | 33.19 | 37.08 | 38.62 |
| SOTS haze (424 valid) | 27.42 | 31.33 | 30.06 |
| Noise s15 | 33.10 | 33.80 (-0.3) | 34.10 |
| Noise s25 | 30.54 | 31.16 (-0.3) | 31.45 |
| Noise s50 | 27.20 | 27.92 (-0.26) | 28.18 |
| Balanced single-task mean3 | 30.30 | 33.12 (-0.19) | 33.31 |
| Synthetic haze + rain | 19.98 | 20.30 (+0.05) | 20.25 |
| Haze + noise | 22.37 | 15.72 | 15.74 |
| Rain + noise | 28.66 | 24.49 | 24.51 |
| CDD-11 haze + rain | 22.02 | 19.79 | 18.67 |

### B8. Students vs released AdaIR (slide 91)
Student work is the student team's; kept here as the reference the teacher has to beat.

| Model | Noise s15 / 25 / 50 | Noise | Rain | Haze (SOTS, unseen scenes) | Mean3 | SOTS training copies | O-HAZE | NH-HAZE |
|---|---|---|---|---|---|---|---|---|
| Released AdaIR | 34.12 / 31.46 / 28.18 | 31.25 | 38.64 | 30.07 (+6.5) | 33.32 | 36.59 | 16.22 | 12.15 |
| S11 | 33.79 / 31.16 / 27.92 | 30.96 | 36.22 | 30.42 (+4.9) | 32.53 | 34.49 | 16.31 | 12.17 |
| S48 | 33.79 / 31.15 / 27.90 | 30.95 | 37.13 | 30.29 (+5.1) | 32.79 | 34.57 | 16.45 | 12.21 |
| S54 | 33.81 / 31.17 / 27.92 | 30.97 | 37.10 | 31.34 (+3.6) | 33.14 | 35.55 | 16.38 | 12.16 |

### Note on "original AdaIR" haze numbers
They differ between tables because the test sets differ: 22.80 (B2, own synthetic haze),
31.05 (B3/B4, held-out haze set), 30.06-30.07 (B7/B8, SOTS outdoor). Compare within a table only.

### A5. Shaped bands + BSA (`patches/shaped_bands.py`), untrained (2026-10-19)
`python tools/check_shaped_bands.py`, figures from `python tools/plot_bands.py`
(`figures/bands_start_masks.png`, `figures/bands_start_kodim.png`).

Masks (random perturbed shapes, 4 samples): min 0, max 1.0000, max |sum - 1| = 1.2e-7 at 16x16, 33x47,
64x96, 256x384. Bands: max |imag| / max |real| <= 1.8e-7, max |sum of bands - P| <= 1.4e-6 (before the
DFT-mirror symmetrisation imag/real was 0.13 at 16x16 and 0.09 at 64x96). BSA at init vs identity: 0.0.
All new parameters get gradients after one Adam step.

Band areas at the starting shapes (r50 = 0.10):

| Map | dot | star | lobe | rest |
|---|---|---|---|---|
| 16x16 | 1.4% | 10.9% | 26.1% | 61.6% |
| 32x32 | 1.4% | 9.1% | 26.5% | 63.0% |
| 64x64 | 1.4% | 8.5% | 26.7% | 63.4% |
| 128x128 to 512x512 | 1.4% | 8.2-8.3% | 26.8% | 63.6% |

(Old v_selfattention_improved fre3, slide 99: 1 / 11 / 29 / 60%.)

Cost (RTX 4080 SUPER, fp32):

| Model | Params | Train step 4x3x128x128 fwd+bwd | Train memory | Inference 1x3x512x768 | Inference memory |
|---|---|---|---|---|---|
| original AdaIR | 28.78M | 239 ms | 12.53 GB | 474 ms | 3.55 GB |
| shaped bands + BSA | 31.13M | 267 ms | 13.15 GB | 510 ms | 3.57 GB |

Real images (Kodak24, untrained model, starting shapes): mean 50%-energy radius r50 and band areas
dot / star / lobe / rest per AFLB:

| Noise sigma | fre1 r50 | fre1 areas | fre2 r50 | fre2 areas | fre3 r50 | fre3 areas |
|---|---|---|---|---|---|---|
| 0 | 0.128 | 2.4 / 7.7 / 26.7 / 63.2% | 0.068 | 1.1 / 8.6 / 26.8 / 63.6% | 0.029 | 0.5 / 9.0 / 26.8 / 63.7% |
| 15 | 0.133 | 2.6 / 7.6 / 26.7 / 63.2% | 0.073 | 1.2 / 8.5 / 26.8 / 63.6% | 0.030 | 0.6 / 9.0 / 26.8 / 63.7% |
| 25 | 0.145 | 3.0 / 7.3 / 26.7 / 63.0% | 0.084 | 1.4 / 8.3 / 26.8 / 63.5% | 0.034 | 0.6 / 9.0 / 26.8 / 63.7% |
| 50 | 0.214 | 5.6 / 6.0 / 26.7 / 61.7% | 0.159 | 3.7 / 6.9 / 26.8 / 62.6% | 0.061 | 1.2 / 8.5 / 26.8 / 63.5% |

The dot follows the image: noise adds high-frequency energy, so the 50% radius grows.

### A6. Bounded mask shapes (2026-10-19)
`python tools/check_shaped_bands.py` (bounded default), `python tools/plot_bounds.py` -> `figures/bands_bounds.png`.

Decoded starting shapes vs INIT_SHAPE: max diff 2.4e-8. Raw SGB outputs in [-50, 50] (plus 1000 random draws
with std 20) all decode inside the bounds:

| Param | Start | Bounds | Space |
|---|---|---|---|
| dot_scale (x r50) | 1.0 | 0.5 - 2.0 | log |
| star_theta | 0 | -22.5 - +22.5 deg | linear |
| star_p | 0.5 | 0.4 - 1.0 (first 0.25: thin-cross collapse, see log) | linear |
| star_size | 0.75 | 0.4 - 0.95 | linear |
| star_aspect | 1.0 | 2/3 - 3/2 | log |
| lobe_vertex | 0.6 | 0.4 - 0.9 | linear |
| lobe_curv | 0.4 | 0.1 - 1.5 | log |
| lobe_theta | 0 | -30 - +30 deg | linear |

Band areas (dot / star / lobe / rest, 128x128, r50 = 0.1) at the extremes, from `figures/bands_bounds.png`:
start 1 / 8 / 27 / 64%; star_p 0.4: 1 / 4 / 27 / 68%; star_p 1.0: 1 / 27 / 25 / 46%; star_size 0.4: 1 / 2 / 27 / 70%;
star_size 0.95: 1 / 14 / 26 / 58%; lobe_vertex 0.4: 1 / 8 / 46 / 45%; lobe_vertex 0.9: 1 / 8 / 4 / 86%;
lobe_curv 0.1: 1 / 8 / 36 / 54%; lobe_curv 1.5: 1 / 8 / 14 / 76%; dot_scale 2: 4 / 6 / 27 / 63%.
All earlier shaped-band checks (A5) unchanged with bounds on.

### A7. Pixel L1 + 0.1 FFT L1: size of the terms (2026-10-19)
`python tools/check_losses.py`. fft_l1 vs SFNet's training code: difference 0.0. Kodak24 random crops
(48 per size); "AdaIR output" = released adair3d on the sigma-25 crops.

| Error | Crop | L_pix | L_fft (unnormalised) | L_fft / L_pix | FFT share of total at w 0.1 | ortho: L_fft / L_pix | ortho share |
|---|---|---|---|---|---|---|---|
| noisy sigma 25 input | 128 | 0.0758 | 6.873 | 90.7 | 90.1% | 0.71 | 6.6% |
| AdaIR output on sigma 25 | 128 | 0.0176 | 1.518 | 86.5 | 89.6% | 0.68 | 6.3% |
| hazy input (t 0.5, A 0.9) | 128 | 0.2376 | 2.012 | 8.5 | 45.9% | 0.07 | 0.7% |
| noisy sigma 25 input | 256 | 0.0757 | 13.725 | 181.4 | 94.8% | 0.71 | 6.6% |
| AdaIR output on sigma 25 | 256 | 0.0180 | 3.201 | 178.1 | 94.7% | 0.70 | 6.5% |
| hazy input (t 0.5, A 0.9) | 256 | 0.2392 | 4.323 | 18.1 | 64.4% | 0.07 | 0.7% |

Under bf16 autocast the loss is fp32 and gradients are finite.

### A8. DCPT functional check (2026-10-19)
`python tools/check_dcpt.py --iters 1500 --batch 2` on the bounded shaped-band AdaIR. Synthetic degradations on
Kodak24 128 crops (noise sigma 15/25/50, haze t 0.3-0.7 A 0.7-1.0, Gaussian blur sigma 1.5-3, low light),
4 classes, images 1-20 train, 21-24 held out. Not a real pre-training run (no datasets yet).

- Classification pass stops after decoder_level1 (refinement calls: 0); features 96x128x128, 96x64x64, 192x32x32.
- L_cls alone gives gradient to the encoder (patch_embed 39.1) and all AFLBs (conv1 2.9e-4 / 2.6e-4 / 6.4e-4,
  fre3 SGB base 4.2e-7; para1 = 0.1).
- Classifier 7.45M params; 435 ms / iteration, 12.5 GB peak (batch 2, fp32).

| Iterations | L_pix (first -> last 10%) | L_cls | train acc | held-out acc (chance 0.25) | held-out per class after (noise / haze / blur / low light) |
|---|---|---|---|---|---|
| 400 | 0.047 -> 0.031 | 0.800 -> 0.408 | 0.33 -> 0.57 | 0.19 -> 0.57 | 0.00 / 0.25 / 0.89 / 1.00 |
| 1500 | 0.039 -> 0.012 | 0.542 -> 0.069 | 0.50 -> 0.89 | 0.19 -> 0.82 | 1.00 / 1.00 / 0.42 / 0.97 |

### A9. fp32 vs bf16 (2026-10-19)
`python tools/check_precision.py`. RTX 4080 SUPER 16 GB; TF32 on for cuDNN convs (PyTorch default), off for matmul.

Training step, shaped-band AdaIR + PixelFFTLoss + AdamW, 128x128 crops:

| Batch | fp32 ms / step | fp32 peak | bf16 ms / step | bf16 peak | bf16 speed-up | ms / image fp32 -> bf16 |
|---|---|---|---|---|---|---|
| 2 | 212 | 6.8 GB | 224 | 3.8 GB | 0.95x | 106 -> 112 |
| 4 | 271 | 13.3 GB | 239 | 7.1 GB | 1.13x | 68 -> 60 |
| 6 | OOM | | 276 | 10.4 GB | | -> 46 |
| 8 | OOM | | 331 | 13.7 GB | | -> 41 |
| 12, 16 | OOM | | OOM | | | |

Best throughput: fp32 batch 4 = 68 ms / image; bf16 batch 8 = 41 ms / image (1.64x).

Released AdaIR (adair3d, frequency step in fp32) inference on Kodak24 sigma 25: fp32 32.534 dB, bf16 32.519 dB;
bf16 - fp32 per image: mean -0.015, min -0.031, max -0.003 dB (worse on every image); max pixel diff 0.0095.
-> evaluate / deploy-reference in fp32.

Shaped-band model, one training step bf16 vs fp32 (same weights, para1 0.1): output rel. diff 3.1e-3,
gradient cosine 0.99998, gradient rel. diff 7.0e-3. Training-quality effect still to be measured with a matched run.

### A10. Evaluation check: released AdaIR (adair3d.ckpt) vs the paper (2026-10-19)
`python eval_fyp.py --ckpt ckpt/adair3d.ckpt --variant original --sets paper3 sots_unseen sots_seen` (protocol adair, fp32).

| Set | Paper Table 1 (PSNR / SSIM) | Ours | Old deck |
|---|---|---|---|
| BSD68 sigma 15 | 34.12 / 0.935 | 34.119 / 0.9353 | 34.12 |
| BSD68 sigma 25 | 31.45 / 0.892 | 31.462 / 0.8921 | 31.46 |
| BSD68 sigma 50 | 28.19 / 0.802 | 28.182 / 0.8019 | 28.18 |
| Rain100L | 38.64 / 0.983 | 38.641 / 0.9830 | 38.64 |
| SOTS outdoor (500) | 31.06 / 0.980 | 31.062 / 0.9802 | |
| SOTS unseen (424) | | 30.072 / 0.9785 | 30.07 |
| SOTS seen in OTS training (76) | | 36.590 / 0.9894 | 36.59 ("SOTS training copies") |

BSD68 noise differs from the paper's draw (seeded per image), so +-0.01 dB there is noise.

### A11. Data checks (2026-10-19)
- SOTS outdoor vs OTS clear (32x32 grey cosine of the best match, 492 gt): [0, 0.8) 156, [0.8, 0.9) 161,
  [0.9, 0.95) 93, [0.95, 0.98) 7, [0.98, 0.999) 0, [0.999, 1] 75. The 75 are pixel-identical -> 76 hazy inputs seen, 424 unseen.
- Fixed mixture test sets: haze_noise25 500 (424 unseen), rain_noise25 100, haze_rain 500 (424 unseen), haze_rain_cdd11 200.
- Training throughput with the real data loader (12 workers), shaped AdaIR, 128 crops, PixelFFTLoss:
  restore fp32 batch 4: 14.5 img/s (13.5 GB); restore bf16 batch 8: 23.3 img/s (13.8 GB); DCPT stage fp32 batch 2:
  4.8 img/s (12.7 GB); original AdaIR fp32 batch 4: 15.3 img/s.
  -> one aio3_mix epoch (159,567 crops): 3.1 h fp32 / 1.9 h bf16; one aio3 epoch (111,567): 2.1 h / 1.3 h.

### A12. True fp32 vs TF32 vs bf16 (2026-10-19)
Shaped AdaIR training step, batch 4, 128 crops, PixelFFTLoss, AdamW: true fp32 (TF32 off) 74.7 ms / image;
TF32 convs + matmuls 61.1 ms; bf16 autocast 58.8 ms (41.4 ms at batch 8, A9). One 3x3 conv (96 -> 96, 64x64) vs a
float64 reference, relative error: fp32 3.2e-7, TF32 3.2e-7 (cuDNN picked a non-TF32 kernel for this shape),
bf16 2.9e-3. Released AdaIR eval in true fp32: BSD68 sigma 25 31.462 / 0.8921, Rain100L 38.641 / 0.9830
(identical to the TF32-default run in A10).
Decision: true fp32 everywhere. aio3_mix epoch ~ 159,567 x 74.7 ms ~ 3.3 h (vs 2.7 h TF32, 1.8 h bf16 batch 8).

### A13. Starting our model from the released backbone (2026-10-19)
First 10 images of each set, protocol adair, true fp32.

| Model | Rain100L | BSD68 sigma 25 | SOTS unseen |
|---|---|---|---|
| released AdaIR | 38.394 | 31.442 | 29.970 |
| released AdaIR, frequency branch off (para1 = 0) | 38.217 | 31.441 | 29.959 |
| ours at init (backbone + para2 from released, fresh AFLBs) | 38.217 | 31.441 | 29.959 |
| ours at init, para2 not carried over (= 1) | 6.094 | 4.186 | 5.644 |

Fresh in ours: 150 tensors (fre1/2/3: sgb, band_ca, bsa, merge, channel_cross_agg, conv1, para1). The released
frequency branch is worth 0.18 dB on Rain100L and ~0.01 dB elsewhere.
Frozen-backbone smoke test (3 frozen + 3 unfrozen steps): backbone tensors changed vs released after step 3: 0 / 497,
after step 6: 497 / 497; trainable 5.00M frozen / 31.13M unfrozen; frozen phase 20 img/s at 8.9 GB (batch 4).

### A14. Our model, training progress (2026-10-20 12:04)
Stage 1 `ours_dcpt` (DCPT warm-up, backbone frozen) finished 07:20 (6.3 h). DCPT top-1 on test images (10 per set):
0.90 at the end (rain100l, sots_unseen, haze_noise25, rain_noise25, haze_rain, haze_rain_cdd11 all 1.00; bsd68_s25
0.30). Train accuracy ~0.94 by step 1,500.
Stage 2 `ours_ft` first in-training validation, step 5,000 = 1.0 epoch, EMA weights, first 10 images per set (PSNR / SSIM):

| Set | ours_ft @ 1.0 epoch | released AdaIR, same 10 images (A13) |
|---|---|---|
| bsd68_s25 | 31.121 / 0.8939 | 31.442 |
| rain100l | 38.648 / 0.9823 | 38.394 |
| sots_unseen | 29.815 / 0.9779 | 29.970 |
| haze_noise25_unseen | 25.691 / 0.8456 | (not measured on these 10; full set 15.74 in B7) |
| rain_noise25 | 30.076 / 0.8624 | (full set 24.51 in B7) |
| haze_rain_unseen | 26.976 / 0.9250 | |
| haze_rain_cdd11 | 26.062 / 0.9443 | (full set 18.67 in B7) |
Early, 10-image numbers only; the comparison that counts is the full eval of both models at 20 epochs.

### A15. Our model, in-training validation through 3 epochs (2026-10-20 22:33)
EMA weights, first 10 images per set, PSNR. Released AdaIR on the same 10 images from A13 where measured.

| Step (epoch) | bsd68_s25 | rain100l | sots_unseen | haze_noise25 | rain_noise25 | haze_rain (ours) | haze_rain_cdd11 |
|---|---|---|---|---|---|---|---|
| 5,000 (1.00) | 31.121 | 38.648 | 29.815 | 25.691 | 30.076 | 26.976 | 26.062 |
| 10,000 (2.01) | 31.274 | 38.709 | 29.926 | 25.940 | 30.230 | 27.607 | 27.594 |
| 15,000 (3.01) | 31.288 | 38.773 | 30.404 | 25.749 | 30.281 | 27.902 | 28.991 |
| 20,000 (4.01) | 31.291 | 38.778 | 30.336 | 25.413 | 30.325 | 27.823 | 29.012 |
| 25,000 (5.01) | 31.365 | 38.821 | 30.155 | 24.922 | 30.327 | 27.631 | 29.420 |
| 30,000 (6.02) | 31.354 | 38.837 | 29.942 | 25.107 | 30.361 | 27.627 | 29.323 |
| 35,000 (7.02) | 31.360 | 38.813 | 30.012 | 25.415 | 30.367 | 27.974 | 29.597 |
| 40,000 (8.02) | 31.335 | 38.827 | 30.326 | 24.962 | 30.380 | 28.275 | 29.861 |
| released AdaIR (A13) | 31.442 | 38.394 | 29.970 | | | | |
Step 18,000 at 22:30 (3.61 epochs), 13.3 img/s, ~55 h left. Paused 12:44-15:45 (GPU share) without loss.

### A16. Are the new frequency blocks used? ours_ft at step 24,000 (4.8 epochs) (2026-10-21 03:00)
EMA weights, 6 images per set, centre 256 crops, CPU. "Off" = para1 = 0 in all three AFLBs (branch output removed).

| Set | Branches on | Branches off | On - off |
|---|---|---|---|
| bsd68_s25 | 31.046 | 31.017 | +0.029 |
| rain100l | 39.929 | 39.915 | +0.013 |
| sots_unseen | 31.813 | 31.802 | +0.011 |
| haze_noise25_unseen | 25.427 | 25.444 | -0.017 |
| rain_noise25 | 30.893 | 30.892 | +0.001 |
| haze_rain_cdd11 | 32.969 | 32.993 | -0.024 |

AFLB change to the decoder feature |agg * para1| / |y * para2|: fre1 0.093, fre2 0.016, fre3 0.047 (released AdaIR:
0.033 / 0.124 / 0.000). para1 mean |.|: 0.0020 / 0.0072 / 0.0060 (released 0.0226 / 0.0380 / 0.0012).
Train loss still falling (total per 5k-step window: 0.1182, 0.1050, 0.1036, 0.1026, 0.1016); band areas moving (fre2 dot
8 -> 14%, lobes wider in all AFLBs). So far the gains over the released model come from fine-tuning the backbone on
the new data / loss, not from the new blocks: the "ignored new blocks" risk (log.md DECISION) is showing.

### A17. Learned mask shapes at 5 epochs (`/home/minura/Himeth/results/mask_shapes/`, 2026-10-21 03:30)
ours_ft step 25,000 (5.01 epochs), EMA weights, 20 test images per degradation, centre 256 crops
(`tools/plot_mask_shapes.py`; figures masks_grid.png + one per degradation, numbers in shape_params.csv).
Band areas dot / star / lobe / rest (%), clean -> CDD-11 haze+rain: fre1 4/8/14/75 -> 13/4/12/71;
fre2 2/8/30/60 -> 9/4/31/56; fre3 1/6/38/56 -> 1/5/38/55.
- fre1 (deepest, 32 px at 256 input) moved most, and several params sit at or near their bounds: star angle
  -20.5 to -21.8 deg (bound -22.5), star aspect 0.70-0.74 (bound 0.667), lobe vertex 0.85-0.88 (bound 0.9),
  lobe curvature 0.101-0.104 (bound 0.1), lobe angle +14 to +18 deg. The lobes are pushed out to the edge.
- fre2 stays close to the start (p 0.50, size 0.75, aspect 1.02, vertex 0.58, curv 0.34, angles ~1 deg).
- fre3 moved a little (p 0.45, size 0.70, aspect 0.87, curv 0.14-0.15, lobe angle 3.5-4 deg).
- Per-degradation adaptation is mostly the dot (it follows the image's 50%-energy radius: fre1 dot_r 0.17 haze,
  0.19 clean / rain, 0.24 noise s50, 0.38 CDD-11 haze+rain) plus 1-3.5 deg angle shifts for haze types; the
  star / lobe shapes are nearly the same for every degradation, i.e. the shape generator adapts very little per image.

### A18. Are the new frequency blocks used? ours_ft at step 50,000 (10.0 epochs) (2026-10-21 20:20)
Same check as A16 (`tools/check_branch_usage.py`, EMA weights, first 6 images per set, centre 256 crops, CPU; numbers
in /home/minura/Himeth/results/branch_usage/step50000.json). "Off" = para1 = 0 in all three AFLBs.

| Set | Branches on | Branches off | On - off | (A16, 4.8 ep) |
|---|---|---|---|---|
| bsd68_s25 | 31.133 | 31.104 | +0.030 | +0.029 |
| rain100l | 40.065 | 40.043 | +0.021 | +0.013 |
| sots_unseen | 31.204 | 31.156 | +0.048 | +0.011 |
| haze_noise25_unseen | 25.570 | 25.580 | -0.010 | -0.017 |
| rain_noise25 | 30.941 | 30.939 | +0.001 | +0.001 |
| haze_rain_cdd11 | 33.307 | 33.528 | **-0.221** | -0.024 |

AFLB change to the decoder feature |agg * para1| / |y * para2|: fre1 0.041, fre2 0.032, fre3 0.049 (A16: 0.093 /
0.016 / 0.047). para1 mean |.|: 0.0019 / 0.0092 / 0.0085 (A16: 0.0020 / 0.0072 / 0.0060). Train loss per 5k-step
window: 0.1182, 0.1050, 0.1036, 0.1026, 0.1015, 0.1018, 0.1004, 0.1000, 0.0995, 0.0990 (still falling slowly). Band
areas from the last training batch (dot / star / lobe / rest): fre1 6 / 0.2 / 12 / 82%, fre2 13 / 2 / 29 / 55%, fre3
2 / 5 / 40 / 54%. The fre1 star has nearly vanished (0.2%).
Reading: after 10 epochs the new blocks still add about +0.03 dB on 4 of 6 sets and cost 0.22 dB on CDD-11 haze+rain
(6 images). The output barely depends on them, and para1 is still ~0.002-0.009. Our gains over the released model so
far come from fine-tuning the backbone (plus the FFT loss / DCPT), which the baseline also gets (except FFT loss / DCPT).
The ours-vs-baseline comparison decides whether ours ends up ahead.

### A19. Ours vs the baseline at matched steps (in-training validation, EMA, first 10 images per set)
Both fine-tune from the released adair3d.ckpt on aio3_mix with the same schedule (32 crops / step). Ours = shaped
bands + DCPT warm-up stage before step 0 + L1 + 0.1 FFT; baseline = original AdaIR, full released weights, L1
(Elgin). Start = the released model (baseline 60-step bench on Elgin; first three sets also measured on qbits:
31.442 / 38.394 / 29.970).

| Step (epochs) | Model | bsd68_s25 | rain100l | sots_unseen | haze_noise25 | rain_noise25 | haze_rain | cdd11 |
|---|---|---|---|---|---|---|---|---|
| 0 | released | 31.44 | 38.40 | 29.96 | 16.01 | 24.61 | 21.06 | 19.34 |
| 5,000 (1.0) | baseline | 31.118 | 38.198 | 29.563 | 25.820 | 29.905 | 26.410 | 25.659 |
| 5,000 (1.0) | ours | 31.121 | 38.648 | 29.815 | 25.691 | 30.076 | 26.976 | 26.062 |
| 5,000 | ours - baseline | +0.003 | **+0.450** | **+0.252** | -0.129 | +0.171 | **+0.566** | **+0.403** |
| 10,000 (2.0) | baseline | 30.812 | 38.261 | 29.681 | 25.659 | 30.057 | 26.502 | 26.686 |
| 10,000 (2.0) | ours | 31.274 | 38.709 | 29.926 | 25.940 | 30.230 | 27.607 | 27.594 |
| 10,000 | ours - baseline | **+0.462** | **+0.448** | **+0.245** | **+0.281** | +0.173 | **+1.105** | **+0.908** |
| 15,000 (3.0) | baseline | 31.149 | 38.306 | 30.091 | 25.855 | 30.120 | 26.980 | 27.621 |
| 15,000 (3.0) | ours | 31.288 | 38.773 | 30.404 | 25.749 | 30.281 | 27.902 | 28.991 |
| 15,000 | ours - baseline | +0.139 | **+0.467** | **+0.313** | -0.106 | +0.161 | **+0.922** | **+1.370** |
| 20,000 (4.0) | baseline | 31.079 | 38.338 | 29.753 | 26.020 | 30.155 | 26.901 | 28.308 |
| 20,000 (4.0) | ours | 31.291 | 38.778 | 30.336 | 25.413 | 30.325 | 27.823 | 29.012 |
| 20,000 | ours - baseline | +0.212 | **+0.440** | **+0.583** | **-0.607** | +0.170 | **+0.922** | **+0.704** |
| 25,000 (5.0) | baseline | 31.234 | 38.364 | 29.549 | 25.133 | 30.186 | 26.767 | 28.419 |
| 25,000 (5.0) | ours | 31.365 | 38.821 | 30.155 | 24.922 | 30.327 | 27.631 | 29.420 |
| 25,000 | ours - baseline | +0.131 | **+0.457** | **+0.606** | -0.211 | +0.141 | **+0.864** | **+1.001** |
| 30,000 (6.0) | baseline | 31.240 | 38.367 | 30.046 | 25.415 | 30.187 | 26.994 | 28.765 |
| 30,000 (6.0) | ours | 31.354 | 38.837 | 29.942 | 25.107 | 30.361 | 27.627 | 29.323 |
| 30,000 | ours - baseline | +0.114 | **+0.470** | -0.104 | **-0.308** | +0.174 | **+0.633** | **+0.558** |
| 35,000 (7.0) | baseline | 31.284 | 38.357 | 29.963 | 24.985 | 30.217 | 26.629 | 29.329 |
| 35,000 (7.0) | ours | 31.360 | 38.813 | 30.012 | 25.415 | 30.367 | 27.974 | 29.597 |
| 35,000 | ours - baseline | +0.076 | **+0.456** | +0.049 | **+0.430** | +0.150 | **+1.345** | +0.268 |
| 40,000 (8.0) | baseline | 31.156 | 38.369 | 29.868 | 25.488 | 30.231 | 26.978 | 29.310 |
| 40,000 (8.0) | ours | 31.335 | 38.827 | 30.326 | 24.962 | 30.380 | 28.275 | 29.861 |
| 40,000 | ours - baseline | +0.179 | **+0.458** | **+0.458** | **-0.526** | +0.149 | **+1.297** | **+0.551** |
| 45,000 (9.0) | baseline | 31.323 | 38.319 | 29.594 | 25.147 | (report truncated) | | |
| 50,000 (10.0) | baseline | 31.256 | 38.353 | 29.700 | 25.428 | 30.244 | 27.214 | 29.727 |
| 50,000 (10.0) | ours | 31.386 | 38.871 | 30.003 | 25.050 | 30.410 | 28.053 | 30.576 |
| 50,000 | ours - baseline | +0.130 | **+0.518** | **+0.303** | **-0.378** | +0.166 | **+0.839** | **+0.849** |
| 55,000 (11.0) | baseline | 31.274 | 38.412 | 29.877 | 25.690 | 30.258 | 27.189 | 30.397 |
| 55,000 (11.0) | ours | 31.377 | 38.871 | 29.989 | 25.053 | 30.416 | 27.944 | 30.770 |
| 55,000 | ours - baseline | +0.103 | **+0.459** | +0.112 | **-0.637** | +0.158 | **+0.755** | **+0.373** |
| 60,000 (12.0) | baseline | 31.349 | 38.404 | 29.877 | 25.099 | 30.266 | 27.127 | 30.087 |
| 60,000 (12.0) | ours | 31.414 | 38.855 | 30.011 | 24.749 | 30.411 | 27.977 | 31.111 |
| 60,000 | ours - baseline | +0.065 | **+0.451** | +0.134 | **-0.350** | +0.145 | **+0.850** | **+1.024** |
| 65,000 (13.0) | baseline | 31.345 | 38.436 | 29.692 | 24.878 | 30.277 | 27.030 | 30.216 |
| 65,000 (13.0) | ours | 31.346 | 38.866 | 29.987 | 25.033 | 30.410 | 28.060 | 30.930 |
| 65,000 | ours - baseline | +0.001 | **+0.430** | **+0.295** | +0.155 | +0.133 | **+1.030** | **+0.714** |

Ours (for later rows): 10k 31.274 / 38.709 / 29.926 / 25.940 / 30.230 / 27.607 / 27.594; 20k 31.291 / 38.778 / 30.336 /
25.413 / 30.325 / 27.823 / 29.012; 30k 31.354 / 38.837 / 29.942 / 25.107 / 30.361 / 27.627 / 29.323; 40k 31.335 /
38.827 / 30.326 / 24.962 / 30.380 / 28.275 / 29.861; 50k 31.386 / 38.871 / 30.003 / 25.050 / 30.410 / 28.053 / 30.576.
At step 5,000 ours is ahead on 5 of 7 sets (rain +0.45, haze +0.25, haze+rain +0.57 / +0.40), equal on noise,
behind on haze+noise (-0.13). The baseline dipped below the released model on the single tasks (-0.3 to -0.4 dB);
ours didn't. Caveats: 10 images per set; ours also had the 1-epoch DCPT warm-up before step 0. Since the new blocks
barely matter (A18), the edge more likely comes from the FFT loss / DCPT stage than from the shaped bands.

### A20. Ours (ours_ft) finished: in-training validation at every epoch (EMA, first 10 images per set) (2026-10-23 07:18)
Finished 99,730 iterations (20.0 epochs) at 2026-10-23 07:18, no non-finite gradient. Weights kept (model + EMA) for
the epochs marked *: runs/ours_ft/iter_0020000.pt, iter_0040000.pt (full) and runs/ours_ft/epochs/step_*.pt;
epochs without * only have these validation numbers. Released model on the same images: 31.442 / 38.394 / 29.970 /
16.01 / 24.61 / 21.06 / 19.34.

| Step | Epoch | bsd68_s25 | rain100l | sots_unseen | haze_noise25 | rain_noise25 | haze_rain | cdd11 | mean 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5,000 | 1.0 | 31.121 | 38.648 | 29.815 | 25.691 | 30.076 | 26.976 | 26.062 | 29.770 |
| 10,000 | 2.0 | 31.274 | 38.709 | 29.926 | 25.940 | 30.230 | 27.607 | 27.594 | 30.183 |
| 15,000 | 3.0 | 31.288 | 38.773 | 30.404 | 25.749 | 30.281 | 27.902 | 28.991 | 30.484 |
| 20,000 | 4.0* | 31.291 | 38.778 | 30.336 | 25.413 | 30.325 | 27.823 | 29.012 | 30.425 |
| 25,000 | 5.0 | 31.365 | 38.821 | 30.155 | 24.922 | 30.327 | 27.631 | 29.420 | 30.377 |
| 30,000 | 6.0 | 31.354 | 38.837 | 29.942 | 25.107 | 30.361 | 27.627 | 29.323 | 30.364 |
| 35,000 | 7.0 | 31.360 | 38.813 | 30.012 | 25.415 | 30.367 | 27.974 | 29.597 | 30.505 |
| 40,000 | 8.0* | 31.335 | 38.827 | 30.326 | 24.962 | 30.380 | 28.275 | 29.861 | 30.567 |
| 45,000 | 9.0 | 31.376 | 38.864 | 30.032 | 25.152 | 30.395 | 28.211 | 30.275 | 30.615 |
| 50,000 | 10.0 | 31.386 | 38.871 | 30.003 | 25.050 | 30.410 | 28.053 | 30.576 | 30.621 |
| 55,000 | 11.0 | 31.377 | 38.871 | 29.989 | 25.053 | 30.416 | 27.944 | 30.770 | 30.631 |
| 60,000 | 12.0* | 31.414 | 38.855 | 30.011 | 24.749 | 30.411 | 27.977 | 31.111 | 30.647 |
| 65,000 | 13.0* | 31.346 | 38.866 | 29.987 | 25.033 | 30.410 | 28.060 | 30.930 | 30.662 |
| 70,000 | 14.0* | 31.421 | 38.863 | 30.209 | 24.557 | 30.420 | 28.213 | 30.873 | 30.651 |
| 75,000 | 15.0* | 31.401 | 38.862 | 30.128 | 24.947 | 30.420 | 28.174 | 30.954 | 30.698 |
| 80,000 | 16.0* | 31.409 | 38.876 | 30.155 | 24.515 | 30.423 | 28.109 | 31.071 | 30.651 |
| 85,000 | 17.0* | 31.405 | 38.888 | 30.249 | 24.576 | 30.425 | 28.142 | 31.182 | 30.695 |
| 90,000 | 18.0* | 31.401 | 38.891 | 30.264 | 24.818 | 30.429 | 28.151 | 31.277 | 30.747 |
| 95,000 | 19.1* | 31.421 | 38.893 | 30.204 | 24.678 | 30.428 | 28.166 | 31.261 | 30.722 |
| 99,730 | 20.0* | 31.426 | 38.885 | 30.194 | 24.648 | 30.426 | 28.156 | 31.328 | 30.723 |

Best mean at epochs 18-20 (~30.72-30.75); flat from epoch 17 on (every set within +-0.15 dB). haze_noise25 peaked at
epoch 2 (25.940) and drifted down to ~24.6-24.8; cdd11 rose all the way (27.6 at epoch 2 -> 31.3).
Single tasks vs the released model (same 10 images), epoch 20: noise s25 -0.02, Rain100L +0.49, SOTS unseen +0.22.

### A21. Full test-set evaluation (the standard numbers for the paper) (2026-10-23)
`eval_fyp.py --sets main` (every image, AdaIR's test protocol: centre crop to a multiple of 16, true fp32, EMA weights
for our runs). PSNR / SSIM. Released = adair3d.ckpt measured here (runs/eval_released/adair3d_main.json); ours =
runs/ours_ft/epochs/step_0099730.pt, epoch 20 (runs/eval_ours_ft/epoch20_main.json). Baseline column to follow.

| Set | n | Paper (AdaIR, ICLR'25 Table 1) | Released (measured) | Ours, epoch 20 | Ours - released |
|---|---|---|---|---|---|
| BSD68 s15 | 68 | 34.12 / 0.935 | 34.119 / 0.9353 | 33.746 / 0.9350 | -0.373 |
| BSD68 s25 | 68 | 31.45 / 0.892 | 31.462 / 0.8921 | 31.152 / 0.8917 | -0.310 |
| BSD68 s50 | 68 | 28.19 / 0.802 | 28.182 / 0.8019 | 27.946 / 0.8029 | -0.236 |
| Rain100L | 100 | 38.64 / 0.983 | 38.641 / 0.9830 | 37.863 / 0.9826 | -0.778 |
| SOTS outdoor (all) | 500 | 31.06 / 0.980 | 31.062 / 0.9802 | 31.220 / 0.9809 | +0.158 |
| SOTS unseen | 424 | | 30.071 / 0.9785 | 30.209 / 0.9793 | +0.138 |
| SOTS seen in OTS | 76 | | 36.590 / 0.9894 | 36.861 / 0.9897 | +0.271 |
| Haze + noise s25 (all) | 500 | | 15.741 / 0.7173 | 24.933 / 0.8438 | +9.192 |
| Haze + noise s25 (unseen) | 424 | | 15.700 / 0.7159 | 24.444 / 0.8397 | +8.744 |
| Rain + noise s25 | 100 | | 24.510 / 0.7374 | 30.744 / 0.8735 | +6.234 |
| Haze + rain (all) | 500 | | 20.457 / 0.8096 | 29.006 / 0.9545 | +8.549 |
| Haze + rain (unseen) | 424 | | 20.336 / 0.8073 | 28.367 / 0.9526 | +8.031 |
| Haze + rain CDD-11 | 200 | | 18.670 / 0.7703 | 29.549 / 0.9615 | +10.879 |

The first-10-image subset reproduces the in-training validation exactly (31.426 / 38.885 / 30.194), but it was not
representative: on all of Rain100L ours is -0.78 dB vs released (+0.49 on the first 10), on BSD68 s25 -0.31 (-0.02).
The 10-image validation tracks trends, not final numbers.

Baseline column added (runs/eval_baseline/epoch6_main.json on Elgin; baseline_ft step 30,000 = epoch 6, EMA; Elgin GPU0,
training paused 50 min for it, no progress lost):

| Set | Paper | Released | Baseline, epoch 6 | Ours, epoch 20 | Ours - baseline |
|---|---|---|---|---|---|
| Noise s15 (BSD68) | 34.12 / 0.935 | 34.119 / 0.9353 | 33.619 / 0.9327 | 33.746 / 0.9350 | +0.127 |
| Noise s25 (BSD68) | 31.45 / 0.892 | 31.462 / 0.8921 | 31.035 / 0.8882 | 31.152 / 0.8917 | +0.117 |
| Noise s50 (BSD68) | 28.19 / 0.802 | 28.182 / 0.8019 | 27.695 / 0.7939 | 27.946 / 0.8029 | +0.251 |
| Rain (Rain100L) | 38.64 / 0.983 | 38.641 / 0.9830 | 37.566 / 0.9812 | 37.863 / 0.9826 | +0.297 |
| Haze (SOTS, all 500) | 31.06 / 0.980 | 31.062 / 0.9802 | 30.865 / 0.9793 | 31.220 / 0.9809 | +0.355 |
| Haze (SOTS, 424 unseen) |  | 30.071 / 0.9785 | 29.848 / 0.9776 | 30.209 / 0.9793 | +0.361 |
| Haze (SOTS, 76 seen in OTS) |  | 36.590 / 0.9894 | 36.540 / 0.9889 | 36.861 / 0.9897 | +0.321 |
| Haze + noise s25 (all 500) |  | 15.741 / 0.7173 | 25.209 / 0.8361 | 24.933 / 0.8438 | -0.276 |
| Haze + noise s25 (424 unseen) |  | 15.700 / 0.7159 | 24.809 / 0.8323 | 24.444 / 0.8397 | -0.365 |
| Rain + noise s25 |  | 24.510 / 0.7374 | 30.313 / 0.8657 | 30.744 / 0.8735 | +0.431 |
| Haze + rain (all 500) |  | 20.457 / 0.8096 | 28.225 / 0.9431 | 29.006 / 0.9545 | +0.781 |
| Haze + rain (424 unseen) |  | 20.336 / 0.8073 | 27.658 / 0.9408 | 28.367 / 0.9526 | +0.709 |
| Haze + rain (CDD-11) |  | 18.670 / 0.7703 | 27.077 / 0.9445 | 29.549 / 0.9615 | +2.472 |
| Avg, paper 5 (s15/25/50, rain, SOTS all) | 32.69 / 0.918 | 32.693 | 32.156 | 32.385 | +0.229 |
| Avg, 4 mixtures (all) | | 19.845 | 27.706 | 28.558 | +0.852 |

Ours (epoch 20) beats the baseline (epoch 6) on 11 of 13 sets: single tasks +0.12..+0.36, rain+noise +0.43, haze+rain
+0.71..+0.78, CDD-11 +2.47; behind only on haze+noise (-0.28 all / -0.37 unseen). The fine-tuned baseline loses more
single-task quality than ours (rain -1.08 vs released, ours -0.78), so that cost comes from multi-degradation
fine-tuning. Caveat: epochs not matched (6 vs 20); ours at epoch 8 is being evaluated, and the baseline continues to 20.

### A22. Matched comparison, full test sets: baseline epoch 6 vs ours epoch 8 and epoch 20 (2026-10-23 12:05)
Same protocol as A21 (every image, AdaIR test protocol, fp32, EMA). Ours epoch 8 = runs/ours_ft/iter_0040000.pt
(runs/eval_ours_ft/epoch8_main.json); epoch 8 is the closest saved epoch of ours to the baseline's 6. PSNR / SSIM;
differences are PSNR (dB) vs the baseline.

| Degradation | Released (measured) | Baseline, epoch 6 | Ours, epoch 8 | Ours ep8 − baseline | Ours, epoch 20 | Ours ep20 − baseline |
|---|---|---|---|---|---|---|
| Noise σ15 (BSD68) | 34.119 / 0.9353 | 33.619 / 0.9327 | 33.624 / 0.9344 | +0.005 | 33.746 / 0.9350 | +0.127 |
| Noise σ25 (BSD68) | 31.462 / 0.8921 | 31.035 / 0.8882 | 31.040 / 0.8912 | +0.005 | 31.152 / 0.8917 | +0.117 |
| Noise σ50 (BSD68) | 28.182 / 0.8019 | 27.695 / 0.7939 | 27.670 / 0.8008 | -0.025 | 27.946 / 0.8029 | +0.251 |
| Rain (Rain100L) | 38.641 / 0.9830 | 37.566 / 0.9812 | 37.862 / 0.9823 | +0.296 | 37.863 / 0.9826 | +0.297 |
| Haze (SOTS, all 500) | 31.062 / 0.9802 | 30.865 / 0.9793 | 31.224 / 0.9807 | +0.359 | 31.220 / 0.9809 | +0.355 |
| Haze + noise σ25 | 15.741 / 0.7173 | 25.209 / 0.8361 | 24.992 / 0.8432 | -0.217 | 24.933 / 0.8438 | -0.276 |
| Rain + noise σ25 | 24.510 / 0.7374 | 30.313 / 0.8657 | 30.659 / 0.8725 | +0.346 | 30.744 / 0.8735 | +0.431 |
| Haze + rain (ours) | 20.457 / 0.8096 | 28.225 / 0.9431 | 28.833 / 0.9528 | +0.608 | 29.006 / 0.9545 | +0.781 |
| Haze + rain (CDD-11) | 18.670 / 0.7703 | 27.077 / 0.9445 | 28.089 / 0.9576 | +1.012 | 29.549 / 0.9615 | +2.472 |
| Average, single tasks (5) | 32.693 | 32.156 | 32.284 | +0.128 | 32.385 | +0.229 |
| Average, mixtures (4) | 19.845 | 27.706 | 28.143 | +0.437 | 28.558 | +0.852 |

At (nearly) matched epochs ours wins 7 of 9 degradations, ties noise s15 / s25 (+0.005) and loses haze+noise
(-0.22 PSNR, but higher SSIM 0.8432 vs 0.8361) and noise s50 by 0.03. Epoch 8 -> 20 adds noise (+0.12..0.28) and
CDD-11 (+1.46); rain and SOTS were already at their final level by epoch 8.

### A23. Exactly matched epochs, full test sets: baseline epoch 8 vs ours epoch 8 (and ours epoch 20) (2026-10-24)
Baseline = Elgin runs/baseline_ft/epochs/step_0040000.pt (EMA), evaluated with the same eval_fyp.py --sets main (log on
Elgin: runs/eval_baseline/epoch8_main.log, 3012 s; summary kept locally in runs/eval_baseline/epoch8_main.json).
Training was paused 21:12-22:02 UTC for it (from step 40,012, resumed automatically).

| Degradation | Released AdaIR (measured) | Baseline, epoch 8 | Ours, epoch 8 | Δ vs baseline | Ours, epoch 20 | Δ vs baseline |
|---|---|---|---|---|---|---|
| Noise σ15 (BSD68) | 34.12 / 0.9353 | 33.49 / 0.9323 | 33.62 / 0.9344 | +0.14 | 33.75 / 0.9350 | +0.26 |
| Noise σ25 (BSD68) | 31.46 / 0.8921 | 31.01 / 0.8883 | 31.04 / 0.8912 | +0.03 | 31.15 / 0.8917 | +0.15 |
| Noise σ50 (BSD68) | 28.18 / 0.8019 | 27.77 / 0.7949 | 27.67 / 0.8008 | -0.10 | 27.95 / 0.8029 | +0.18 |
| Rain (Rain100L) | 38.64 / 0.9830 | 37.64 / 0.9815 | 37.86 / 0.9823 | +0.22 | 37.86 / 0.9826 | +0.23 |
| Haze (SOTS outdoor) | 31.06 / 0.9802 | 30.90 / 0.9795 | 31.22 / 0.9807 | +0.33 | 31.22 / 0.9809 | +0.32 |
| Haze + noise σ25 | 15.74 / 0.7173 | 25.14 / 0.8364 | 24.99 / 0.8432 | -0.15 | 24.93 / 0.8438 | -0.21 |
| Rain + noise σ25 | 24.51 / 0.7374 | 30.40 / 0.8671 | 30.66 / 0.8725 | +0.26 | 30.74 / 0.8735 | +0.34 |
| Haze + rain | 20.46 / 0.8096 | 28.18 / 0.9441 | 28.83 / 0.9528 | +0.65 | 29.01 / 0.9545 | +0.82 |
| Haze + rain (CDD-11) | 18.67 / 0.7703 | 27.78 / 0.9479 | 28.09 / 0.9576 | +0.31 | 29.55 / 0.9615 | +1.76 |
| Average, single (5) | 32.69 | 32.16 | 32.28 | +0.12 | 32.39 | +0.23 |
| Average, mixed (4) | 19.84 | 27.88 | 28.14 | +0.27 | 28.56 | +0.68 |

Other baseline epoch-8 sets: SOTS unseen 29.894 / 0.9778 (ours ep8 30.204), SOTS seen 36.497 / 0.9889 (36.912),
haze+noise unseen 24.720 / 0.8325 (24.535), haze+rain unseen 27.600 / 0.9419 (28.208).
At the same epoch (8 vs 8) ours wins 7 of 9 degradations (rain +0.22, SOTS +0.33, rain+noise +0.26, haze+rain +0.65,
CDD-11 +0.31, noise s15 +0.14, s25 +0.03) and loses noise s50 (-0.10) and haze+noise (-0.15 PSNR, but higher SSIM
0.843 vs 0.836). From epoch 6 to 8 the baseline lost a little on noise s15 (-0.13) and gained on CDD-11 (+0.71), as the
10-image validation suggested; the CDD-11 gap narrowed from +1.01 to +0.31.
