# PRISM-IR teacher: answers for the CVPR paper

These are answers to the questions about the teacher runs. Every number is copied from the saved result files listed in
section 6.3 (`results.md` A10–A30 has the same numbers with full detail). Anything we do not have is marked
**not available**, with what it would take to get it.

---

## The most important point: there are two teacher models, not one

The questions read as if the CDD-11 classes were added to the all-in-one run. **They were not.** We trained two
separate models with the same architecture:

| | **Model A: all-in-one** | **Model B: CDD-11** |
|---|---|---|
| Trained on | AIO-3 (noise, rain, haze) + 4 mixtures | 6 CDD-11 classes only |
| Starts from | released AdaIR **3-task** weights | released AdaIR **5-task** weights (it has seen low light) |
| Training samples per epoch | 159,567 | 6,858 |
| Weights | `runs/ours_ft/epochs/step_0099730.pt` | `runs/cdd6_ft384/epochs/epoch_05.pt` |
| Use its numbers for | AIO-3, synthetic mixtures | the CDD-11 table |

- Model A has **never seen low-light images**, and it was tested on only one CDD-11 class (haze+rain).
- Model B was **not tested on AIO-3**.
- In the paper, the CDD-11 table and the AIO-3 table therefore come from different weights. The paper should say so,
  as OneRestore and other CDD-11 papers do when they train a separate model per benchmark.

---

## 1. What the runs were trained on

### 1.1 The CDD-11 class list (Model B)

**Trained and tested on exactly 6 classes: `low`, `haze`, `rain`, `haze+rain`, `low+haze`, `low+rain`.**

This is **3 single + 3 two-way mixes**, not "6 single + 3 mixed". Left out (never trained or tested):

| Left out | Why |
|---|---|
| `snow`, `low+snow`, `haze+snow`, `low+haze+snow` | all snow classes |
| `low+haze+rain` | the 3-way mix. It has no snow, but it was **also left out** |

So for OneRestore, MoCE-IR-S and BioIR, **average their per-class results over these same 6 classes**. We already did
this for two methods from EvoIR's Table 4 (PSNR / SSIM):

| Method | 6-class mean |
|---|---|
| EvoIR | 29.17 / 0.898 |
| OneRestore | 28.96 / 0.893 |
| WGWSNet | 27.00 / 0.874 |
| PromptIR | 26.34 / 0.867 |

MoCE-IR-S and BioIR are **not computed yet**: take their per-class CDD-11 tables and average the same 6 columns. Those
methods were trained on all 11 classes and ours on 6, which the paper should state next to the table.

### 1.2 How much of the CDD-11 training split was used (Model B)

- CDD-11 has **1,183 training scenes per class**. We used **1,143 per class**.
- **40 scenes are held out** (`splits/cdd11_val_scenes.txt`, chosen with seed 0). They are the same 40 scenes in every
  class, so no held-out scene is trained on through another class. They are used only for validation, for the loss
  weight search and for choosing the test tile size. **The test split is never used to choose anything.**
- Each epoch uses every training pair once: **6 × 1,143 = 6,858 samples, exactly balanced** between the classes.
- No synthetic mixtures and no other dataset are in Model B.

### 1.3 Data per epoch (Model A)

**Unchanged: 159,567 samples per epoch, as in the write-up.** Adding Model B did not change Model A.

| Source | Samples per epoch | How it is made |
|---|---|---|
| Noise | 15,432 | BSD400 + WED clean (5,144 images × 3), Gaussian noise σ ∈ {15, 25, 50} added on the fly |
| Rain (Rain100L) | 24,000 | the 200 Rain100L training pairs × 120 |
| Haze (OTS) | 72,135 | RESIDE OTS hazy → clear |
| Haze + noise | 12,000 | OTS hazy + noise σ ∈ {15, 25, 50} |
| Rain + noise | 12,000 | Rain100L rainy + noise σ ∈ {15, 25, 50} |
| Haze + rain (synthetic) | 12,000 | OTS hazy + the rain-streak layer of a random Rain100L pair |
| Haze + rain (CDD-11) | 12,000 | CDD-11 train `haze_rain` (all 1,183 pairs, drawn 12,000 times) |
| **Total** | **159,567** | |

The only CDD-11 data in Model A is the haze+rain class.

### 1.4 Rain data

**Rain100L only:** the 200 training pairs in AdaIR's `rainTrain.txt`. **No Rain200L.**

---

## 2. Is the recipe the same as the write-up?

### 2.1 Model A: yes, apart from small details the write-up leaves out

| Setting | Write-up | Actual | Same? |
|---|---|---|---|
| DCPT warm-up | 5k iterations at 2e-4 | 5,000 iterations, lr 2e-4 (the classifier 5e-5), batch 2 × 16, weight decay 0, backbone frozen | ✔ (details added) |
| Fine-tuning length | 99,730 iterations | 99,730 (= 20 epochs) | ✔ |
| Batch | 32 | 32 (4 crops × 8 accumulated steps) | ✔ |
| Crop size | 128² | 128² | ✔ |
| Optimiser | AdamW | AdamW, betas 0.9 / 0.999 | ✔ |
| Learning rate | cosine 1e-4 → 1e-6 | cosine 1e-4 → 1e-6, **with a 2% linear warm-up** | ✔ (warm-up added) |
| Weight decay | 0.01 | 0.01 | ✔ |
| EMA | 0.999 | 0.999; the EMA weights are the ones tested | ✔ |
| Loss | L1 + 0.1 × frequency-domain L1 | L1 + 0.1 × L1 between the FFTs (unnormalised) | ✔ |
| Backbone init | 3-task AdaIR | released `adair3d.ckpt` backbone (+ `para2`); the new frequency blocks start fresh | ✔ |
| Augmentation | – | random crop + one random flip / rotation | (add) |
| Precision | – | true fp32 (TF32 off) | (add) |

### 2.2 The 6 DCPT classes

| Model | The 6 DCPT classes |
|---|---|
| A | noise, rain, haze, haze+noise, rain+noise, haze+rain |
| B | low, haze, rain, haze+rain, low+haze, low+rain |

In Model A, the three noise levels share the `noise` class. The synthetic haze+rain and CDD-11 haze+rain also share one
class (`haze+rain`).

DCPT accuracy:
- **Model A: 0.90** on test images (10 per set). Every set scores 1.00 except BSD68 σ25 (0.30).
- **Model B: 0.85** on held-out scenes (10 per class).

### 2.3 Model B uses a different recipe

| | DCPT | Stage 1 | Stage 2 | Stage 3 (final) |
|---|---|---|---|---|
| Starts from | released 5-task backbone | DCPT | stage 1 | stage 2 |
| Crop size | 128 | 128 | 256 | 384 |
| Length | 2,000 iterations | 2,246 steps (10.5 epochs) | 2,140 steps (10 epochs) | 1,070 steps (5 epochs) |
| Batch | 2 × 16 | 4 × 8 | 1 × 32 | 1 × 32 |
| Learning rate | 2e-4 | 1e-4, cosine (stopped at about 5e-5) | 5e-5 → 1e-6 | 3e-5 → 1e-6 |
| Loss | clean L1 + classification | mix loss + search | mix loss + search | mix loss + search |

Things that also differ from Model A:

- **Loss:** a × L1 + (1 − a) × (1 − MS-SSIM) + 0.1 × FFT-L1, after EvoIR.
  - At the start of each epoch, 3 values of `a` each train a copy of the model for 20 steps.
  - Each copy is scored on held-out crops, and the best one is kept.
  - `a` stayed between 0.64 and 1.0, and ended at 0.866.
- **EMA:** 0.999, with a warm-up (decay = min(0.999, (1+t)/(10+t)) at step t), because the runs are short.
- **Memory:** gradient checkpointing in stage 3, so the 384-px crops fit on the 16 GB GPU.
- **Everything else is the same as Model A:** AdamW, weight decay 0.01, 2% warm-up, true fp32.

### 2.4 Training time and GPU

All of our runs used **one NVIDIA RTX 4080 SUPER (16 GB)** on our server (qbits).

| Run | Time |
|---|---|
| Model A, DCPT | 6.3 h |
| Model A, fine-tuning | about 67 h of training (13.3 img/s); 3 days wall-clock with two pauses |
| **Model A, total** | **about 73 GPU-hours** |
| Model B, all stages | about 36 h wall-clock (Nov 7 17:30 → Nov 9 05:45), **including** the full CDD-11 test after every epoch and short checks between stages |
| Fine-tuned AdaIR baseline (for Model A) | about 6 days (Oct 21 → Oct 27) on one Quadro RTX 5000, a slower GPU, on a second server |

---

## 3. Results (EMA weights)

### 3.1 CDD-11, Model B (all 200 test images per class)

PSNR / SSIM:

| Method | low | haze | rain | haze+rain | low+haze | low+rain | **6-class mean** |
|---|---|---|---|---|---|---|---|
| **Ours** | 26.14 / 0.8294 | 33.07 / 0.9897 | 33.12 / 0.9626 | 29.43 / 0.9540 | 24.81 / 0.8169 | 24.85 / 0.7899 | **28.57 / 0.8904** |
| Ours+ (×8 self-ensemble) | 26.17 / 0.8308 | 33.35 / 0.9903 | 33.35 / 0.9647 | 29.75 / 0.9573 | 24.90 / 0.8192 | 24.95 / 0.7940 | 28.75 / 0.8927 |
| OneRestore (published) | 26.48 / 0.826 | 32.52 / 0.990 | 33.40 / 0.964 | 29.99 / 0.957 | 25.79 / 0.822 | 25.58 / 0.799 | 28.96 / 0.893 |
| EvoIR (published) | 27.06 / 0.830 | 32.24 / 0.991 | 34.03 / 0.970 | 29.76 / 0.965 | 25.92 / 0.824 | 26.02 / 0.806 | 29.17 / 0.898 |

- **"Ours" is the fair comparison.** The other methods do not use the self-ensemble, so "Ours+" should only appear
  marked with "+".
- Ours is the best of the three on haze. The gap is in the low-light mixes.
- The test is run in 512-px tiles (section 3.3). That size was chosen on the held-out training scenes, not on the
  test.
- Model A on CDD-11 haze+rain only (AdaIR protocol, whole image): **29.55 / 0.9615**.

### 3.2 AIO-3, Model A (epoch 20, every test image)

PSNR / SSIM. "Released AdaIR" is the released checkpoint measured with our code. "Baseline" is the original AdaIR
fine-tuned with the same data, schedule and EMA as ours (L1 loss, no DCPT).

| Test set | n | Released AdaIR | Baseline (epoch 20) | **Ours (epoch 20)** |
|---|---|---|---|---|
| BSD68 σ15 | 68 | 34.12 / 0.9353 | 33.65 / 0.9333 | 33.75 / 0.9350 |
| BSD68 σ25 | 68 | 31.46 / 0.8921 | 31.05 / 0.8890 | 31.15 / 0.8917 |
| BSD68 σ50 | 68 | 28.18 / 0.8019 | 27.84 / 0.7976 | 27.95 / 0.8029 |
| Rain100L | 100 | 38.64 / 0.9830 | 37.83 / 0.9824 | 37.86 / 0.9826 |
| SOTS outdoor, all | 500 | 31.06 / 0.9802 | 31.00 / 0.9800 | 31.22 / 0.9809 |
| SOTS outdoor, 424 subset | 424 | 30.07 / 0.9785 | 30.00 / 0.9783 | 30.21 / 0.9793 |
| **Average (SOTS all 500)** | | 32.69 / 0.9185 | 32.27 / 0.9165 | **32.39 / 0.9186** |
| Average (SOTS 424 subset) | | 32.50 / 0.9182 | 32.07 / 0.9161 | 32.18 / 0.9183 |

- **SOTS:**
  - The main number uses **all 500 images**, as in AdaIR's Table 1.
  - We also report the **424 subset**: 76 of the 500 hazy SOTS inputs are pixel-identical to OTS training images, so
    those 424 are the images the models never saw.
- **Released AdaIR:** our code reproduces AdaIR's Table 1 (for example Rain100L 38.64 / 0.983, SOTS 31.06 / 0.980).
- **Cost of training on mixtures:** both fine-tuned models lose a little on noise and rain compared with the released
  model. Ours loses less than the baseline.

### 3.3 How PSNR and SSIM are computed

| | AIO-3 and mixtures (Model A) | CDD-11 (Model B) |
|---|---|---|
| Protocol | AdaIR's `test.py` | OneRestore's (`img_file/cal_psnr_ssim.py`) |
| Colour | **RGB** (not Y) | **RGB** (not Y) |
| Border crop | **none** | **none** |
| Image size | full image, centre-cropped to a multiple of 16 (as AdaIR) | full image (padded for the network, cropped back) |
| Clipping | output clipped to [0, 1] | output clipped to [0, 1], then **rounded to 8 bits** (as when saved to PNG) |
| How the model is run | whole image at once | 512 × 512 tiles, 128-px overlap, blended |
| PSNR | `skimage` `peak_signal_noise_ratio`, data_range 1 | same |
| SSIM | `skimage` `structural_similarity`, default 7 × 7 window, data_range 1, RGB channels | same |
| Precision | true fp32 | TF32 (checked against fp32: −0.0004 dB) |
| Weights | EMA | EMA |
| Noise | added after cropping, with a fixed seed per image | – |

Every number is computed per image on the full image (not on crops), then averaged over the test set.

### 3.4 Synthetic and single-task mixture tests (Model A)

PSNR / SSIM:

| Test set | n | Released AdaIR | Baseline (epoch 20) | **Ours (epoch 20)** |
|---|---|---|---|---|
| Haze + noise σ25 | 500 | 15.74 / 0.7173 | 25.01 / 0.8382 | 24.93 / 0.8438 |
| Haze + noise σ25 (424 subset) | 424 | 15.70 / 0.7159 | 24.54 / 0.8339 | 24.44 / 0.8397 |
| Rain + noise σ25 | 100 | 24.51 / 0.7374 | 30.52 / 0.8689 | 30.74 / 0.8735 |
| Haze + rain (synthetic) | 500 | 20.46 / 0.8096 | 28.55 / 0.9475 | 29.01 / 0.9545 |
| Haze + rain (synthetic, 424 subset) | 424 | 20.34 / 0.8073 | 27.95 / 0.9453 | 28.37 / 0.9526 |
| Haze + rain (CDD-11 test) | 200 | 18.67 / 0.7703 | 28.88 / 0.9528 | 29.55 / 0.9615 |
| **Average, 4 mixtures (all images)** | | 19.84 / 0.7587 | 28.24 / 0.9019 | **28.56 / 0.9083** |

- The mixture test sets were made once with fixed seeds and saved, so every model sees the same inputs.
- Ours beats the baseline on 8 of 9 test sets in PSNR and on all 9 in SSIM (the 5 single-task sets in 3.2 and the 4
  mixtures here). The only PSNR loss is haze+noise (−0.08), where our SSIM is higher.

### 3.5 AdaIR evaluated with the same code

| Benchmark | AdaIR with the same code? |
|---|---|
| AIO-3 and mixtures | **Yes**, two versions in the tables above: the released checkpoint, and the original AdaIR fine-tuned with our exact data and schedule (the fair baseline) |
| CDD-11, 6 classes | **Not available.** Neither the released AdaIR nor an AdaIR trained on these 6 classes has been tested |

For CDD-11 there are two options. Each needs your OK:

1. Test the released 5-task AdaIR as it is (about 30–40 min). It has never seen most of these classes, so it would be a
   weak reference.
2. Train the original AdaIR on the 6 classes with Model B's recipe, which is the fair baseline. This takes about the
   same time as Model B (about 1.5 days).

---

## 4. Ablations

There is **no clean, one-change-at-a-time ablation** from the current runs. Here is what exists.

**From the current runs:**

1. **Full method vs the fine-tuned original AdaIR** (same data, schedule and EMA; section 3.2):
   - single tasks +0.11 dB;
   - mixtures +0.32 dB;
   - better on 8 of 9 sets in PSNR and on all 9 in SSIM.

   This changes the bands, band attention, FFT loss and DCPT **all together**, so it cannot be split between them.
2. **Frequency blocks switched off at test time** (Model A at 5 and 10 epochs, 6 images per set, 256-px crops):
   - Setting the block outputs to zero changed PSNR by **−0.22 to +0.05 dB**.
   - At those points the output depended only a little on the new blocks.
   - This check was **not repeated on the final (epoch 20) weights**.

   This matters for the paper: without a real ablation, the size of the gain from the shaped bands themselves is not
   known.
3. **CDD-11 crop-size and tile-size steps** (Model B). These are training and test choices, not architecture:

   | Step | 6-class mean |
   |---|---|
   | 128-px crops, whole image | 26.32 |
   | 128-px crops, 256-px tiles | 27.10 |
   | 256-px crops, 256-px tiles | 27.63 |
   | 256-px crops, 512-px tiles | 28.27 |
   | 384-px crops, 512-px tiles | 28.57 |

**From the old server** (from the feasibility slides). Those runs and weights were lost. The setup and test sets were
different, so these numbers **cannot be mixed with the tables above**.

- **2-band rectangle vs 4 shaped bands:**

  | | Single-task mean | Mixture mean |
  |---|---|---|
  | Rectangle | 32.41 | 22.09 |
  | 4 shaped bands | 32.45 | 22.10 |

  This is about a tie.
- **Shape learning rate ×10 / ×100:** within ±0.1 dB.
- **9-step ablation of AdaIR's frequency module:**
  - removing the L-H gate cost 2.6 dB on haze+noise;
  - removing the H-L gate cost 0.3 dB;
  - turning the soft rectangle mask on added 0.39 dB on haze+noise;
  - the other variants were within about ±0.3 dB.

**Not available at all:**
- with / without band self-attention (BSA);
- with / without DCPT.

A clean ablation would be about 4 short runs on CDD-11 (one change each, about 1–1.5 days each on our GPU). Not
started; it needs your OK.

---

## 5. Distilling student 2

**Not done in this repository.** No student has been trained or distilled here, so the student-2 numbers are **not
available**.

The friend's CDD-11 students on the second server (S70-CDD11-W16 / W24) are a different thing: they were trained from
scratch on ground truth, with no teacher.

To help with the design, these are the teacher's feature levels:

| Level | Size | Channels | Where it is |
|---|---|---|---|
| Encoder 1 | 1 | 48 | before the first down |
| Encoder 2 | 1/2 | 96 | |
| Encoder 3 | 1/4 | 192 | |
| **Latent (deepest)** | **1/8** | **384** | frequency block 1 works here |
| Decoder 3 | 1/4 | 192 | frequency block 2 works here |
| Decoder 2 | 1/2 | 96 | frequency block 3 works here |
| Decoder 1 + refinement | 1 | 96 | |
| Output | 1 | 3 | residual added to the input |

There is **no teacher feature at 1/16**. There are two ways to handle the student's 1/16 bottleneck:

1. **Skip it (my suggestion).** Match the output and the 1/2, 1/4 and 1/8 levels, where both networks have features.
   Pooling the teacher down to 1/16 would invent a target the teacher never computed.
2. **Pool the teacher's 1/8 latent by 2 × 2 and add a 1 × 1 adapter.** This reuses the bottleneck adapter, but the
   target is only an average of the teacher's features.

This is the student team's call.

Which teacher to use:
- **Model A** for AIO-3 and the synthetic mixtures.
- **Model B** for CDD-11. On training crops (128–384 px) it can run whole, without tiles.

---

## 6. Figure material

### 6.1 Learned shape parameters (real values)

**Where they come from:**
- **Model A at epoch 20.**
- 20 test images per degradation, centre 256 × 256 crop, mean over the 20 images.
- The spread between images is small, except for the dot. The dot radius varies by up to ±0.13 (standard deviation)
  between images; the other shape values by at most 0.03 and the angles by at most about 1°.
- Source files: `/home/minura/Himeth/results/meeting_10_25/mask_shapes_epoch20/shape_params.csv` (and `.json`).

**Symbols:**
- The symbols are the paper's: s, θ_s, p, c, κ, v₀, k, θ_l.
- The dot is given as its radius **r = s · r50** (a fraction of the highest frequency), because only r was saved, not
  s. r50 is the radius that holds half of each image's energy.
- Angles are in degrees.
- fre1 is the deepest block (1/8 size), fre3 the shallowest (1/2 size).

**fre1**

| Degradation | r = s·r50 | θ_s (°) | p | c | κ | v₀ | k | θ_l (°) | dot / star / lobe / rest (%) |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.095 | -22.4 | 0.403 | 0.417 | 0.680 | 0.784 | 0.100 | -6.0 | 1.4 / 1.0 / 18.3 / 79.3 |
| noise σ25 | 0.100 | -22.4 | 0.403 | 0.418 | 0.681 | 0.783 | 0.100 | -6.0 | 1.5 / 1.0 / 18.4 / 79.2 |
| noise σ50 | 0.115 | -22.4 | 0.403 | 0.417 | 0.680 | 0.784 | 0.100 | -6.0 | 1.8 / 0.9 / 18.3 / 79.0 |
| rain | 0.088 | -22.4 | 0.402 | 0.415 | 0.678 | 0.788 | 0.100 | -6.1 | 1.3 / 1.0 / 17.9 / 79.7 |
| haze | 0.075 | -22.5 | 0.401 | 0.404 | 0.670 | 0.822 | 0.100 | -7.7 | 1.1 / 1.0 / 14.8 / 83.1 |
| haze+noise σ25 | 0.080 | -22.5 | 0.401 | 0.404 | 0.671 | 0.822 | 0.100 | -7.7 | 1.2 / 1.0 / 14.8 / 83.0 |
| rain+noise σ25 | 0.093 | -22.4 | 0.402 | 0.415 | 0.678 | 0.788 | 0.100 | -6.1 | 1.4 / 1.0 / 17.9 / 79.7 |
| haze+rain (CDD-11) | 0.163 | -22.5 | 0.401 | 0.403 | 0.670 | 0.829 | 0.100 | -8.1 | 2.9 / 0.6 / 14.2 / 82.3 |

**fre2**

| Degradation | r = s·r50 | θ_s (°) | p | c | κ | v₀ | k | θ_l (°) | dot / star / lobe / rest (%) |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.128 | 1.9 | 0.499 | 0.738 | 1.001 | 0.592 | 0.340 | 0.7 | 2.3 / 7.1 / 29.4 / 61.1 |
| noise σ25 | 0.146 | 1.9 | 0.499 | 0.738 | 1.001 | 0.592 | 0.340 | 0.7 | 2.9 / 6.8 / 29.4 / 60.9 |
| noise σ50 | 0.203 | 1.9 | 0.499 | 0.738 | 1.001 | 0.592 | 0.340 | 0.7 | 5.1 / 5.7 / 29.4 / 59.8 |
| rain | 0.131 | 2.0 | 0.499 | 0.737 | 1.001 | 0.592 | 0.339 | 0.8 | 2.5 / 7.0 / 29.5 / 61.1 |
| haze | 0.096 | 2.2 | 0.498 | 0.736 | 1.000 | 0.590 | 0.330 | 0.9 | 1.6 / 7.6 / 29.9 / 60.9 |
| haze+noise σ25 | 0.123 | 2.2 | 0.498 | 0.736 | 1.000 | 0.590 | 0.330 | 0.9 | 2.4 / 7.1 / 29.9 / 60.6 |
| rain+noise σ25 | 0.154 | 2.0 | 0.499 | 0.737 | 1.001 | 0.592 | 0.339 | 0.8 | 3.4 / 6.5 / 29.5 / 60.6 |
| haze+rain (CDD-11) | 0.321 | 2.4 | 0.497 | 0.735 | 1.000 | 0.589 | 0.326 | 1.0 | 9.7 / 3.3 / 30.0 / 57.0 |

**fre3**

| Degradation | r = s·r50 | θ_s (°) | p | c | κ | v₀ | k | θ_l (°) | dot / star / lobe / rest (%) |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.030 | 0.5 | 0.446 | 0.692 | 0.840 | 0.567 | 0.120 | 7.6 | 0.5 / 5.6 / 38.6 / 55.3 |
| noise σ25 | 0.031 | 0.5 | 0.446 | 0.693 | 0.840 | 0.567 | 0.120 | 7.6 | 0.6 / 5.6 / 38.5 / 55.3 |
| noise σ50 | 0.036 | 0.5 | 0.445 | 0.693 | 0.840 | 0.567 | 0.120 | 7.6 | 0.6 / 5.5 / 38.6 / 55.3 |
| rain | 0.024 | 0.5 | 0.445 | 0.692 | 0.838 | 0.567 | 0.118 | 7.7 | 0.5 / 5.6 / 38.6 / 55.3 |
| haze | 0.017 | 0.6 | 0.439 | 0.683 | 0.820 | 0.562 | 0.112 | 8.8 | 0.4 / 5.3 / 39.2 / 55.1 |
| haze+noise σ25 | 0.019 | 0.6 | 0.439 | 0.683 | 0.820 | 0.562 | 0.111 | 8.8 | 0.5 / 5.2 / 39.2 / 55.1 |
| rain+noise σ25 | 0.027 | 0.5 | 0.445 | 0.692 | 0.838 | 0.567 | 0.118 | 7.7 | 0.5 / 5.5 / 38.6 / 55.3 |
| haze+rain (CDD-11) | 0.058 | 0.6 | 0.440 | 0.685 | 0.822 | 0.563 | 0.112 | 8.6 | 0.9 / 4.9 / 39.1 / 55.1 |

For comparison, the starting values (before training) are s 1.0, θ_s 0°, p 0.5, c 0.75, κ 1.0, v₀ 0.6, k 0.4 and
θ_l 0°. The bands then cover about 1 / 8 / 27 / 64%.

**What the real values show** (the figure should match this):

- **The dot adapts most.** In fre2 it grows from 1.6% of the spectrum (haze) to 5.1% (noise σ50) and 9.7% (CDD-11
  haze+rain).
- **Star and lobes change little between degradations:** shape values by at most about 0.05, angles by at most about
  2°. The largest changes are in fre1 for haze-type inputs: the lobes start further out (v₀ 0.78 → 0.83) and
  rotate about 2° more.
- **In fre1, several values sit at their limits:**
  - θ_s −22.4° (limit −22.5°);
  - p 0.40 (limit 0.4);
  - κ 0.67 (limit 0.667);
  - k 0.100 (limit 0.1).

  The fre1 star shrank to about 1% of the spectrum.
- **A figure with very different shapes per degradation would overstate this.** Real values show one shape per block
  that changes mainly in dot size.

**Not available:**
- **Per-image values for single chosen images:** only means and spreads were saved. Getting them is a short CPU run.
- **Model B's learned shapes:** never extracted.

Both need your OK.

**Ready-made figures:**
- `results/meeting_10_25/4_learned_mask_shapes_epoch20.png` (all blocks × degradations);
- one figure per degradation in `results/meeting_10_25/mask_shapes_epoch20/`.

### 6.2 Visual examples

| | Status |
|---|---|
| **Model A** | `/home/minura/Himeth/results/meeting_10_25/3_visual_examples_mixtures.png`: **input / released AdaIR / ours / ground truth** on 4 composite degradations (haze+noise, rain+noise, synthetic haze+rain, CDD-11 haze+rain). The rows are not hand-picked: each is the test image whose gain is closest to that set's median gain. Made by `tools/visual_examples.py` |
| **Model B** (CDD-11 composites: low+haze, low+rain, haze+rain) | **Not available.** No output images were saved, only the scores. They can be made with the same median-gain rule in a short run, if you OK it |

### 6.3 Checkpoints and evaluation scripts

Everything is on qbits under `/home/minura/Himeth/AdaIR/`. Use the **`ema`** weights inside each checkpoint.

| What | Path |
|---|---|
| Model A, final (epoch 20) | `runs/ours_ft/epochs/step_0099730.pt` |
| Model B, final (stage 3, epoch 5) | `runs/cdd6_ft384/epochs/epoch_05.pt` |
| Fine-tuned AdaIR baseline (epoch 20) | second server: `/mnt/minura/minura/Himeth/AdaIR/runs/baseline_ft/epochs/step_0099730.pt` |
| Released AdaIR | `ckpt/adair3d.ckpt` (3-task), `ckpt/adair5d.ckpt` (5-task) |
| Saved scores, Model A / baseline / released | `runs/eval_ours_ft/epoch20_main.json`, `runs/eval_baseline/epoch20_main.json`, `runs/eval_released/adair3d_main.json` |
| Saved scores, Model B | `runs/cdd6_ft384/test/epoch_05.json` (no flips), `runs/cdd6_final/test_ep5_tile512.json` (×8) |
| Test code | `fyp/evaluate.py` (test sets, protocols, tiles, self-ensemble), `fyp/metrics.py` (PSNR / SSIM) |

Commands to re-run the numbers (from the `AdaIR` folder, after `source /home/minura/Himeth/env.sh`):

```bash
# Model A: AIO-3 + mixtures, AdaIR protocol, true fp32
python eval_fyp.py --ckpt runs/ours_ft/epochs/step_0099730.pt --ema --sets main --out runs/eval_ours_ft/epoch20_main.json
```

```bash
# Released AdaIR, same sets and code
python eval_fyp.py --ckpt ckpt/adair3d.ckpt --variant original --sets main --out runs/eval_released/adair3d_main.json
```

```bash
# Model B: CDD-11, OneRestore protocol, 512-px tiles, without and with x8 (uses the EMA weights)
python tools/cdd_infer_sweep.py --ckpt runs/cdd6_ft384/epochs/epoch_05.pt --sets cdd6 --settings tile512 tile512_x8 --out runs/cdd6_final/rerun.json
```

On the RTX 4080 SUPER:

| Run | Time |
|---|---|
| Model A test | about 30 min |
| Model B, no flips | about 30 min |
| Model B, ×8 | about 4 h |
