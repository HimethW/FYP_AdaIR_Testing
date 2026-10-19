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
