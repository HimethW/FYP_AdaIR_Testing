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
