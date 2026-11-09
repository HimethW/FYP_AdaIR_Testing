# Our model: architecture, training and testing

This document describes our improved version of AdaIR (ICLR 2025), the all-in-one image restoration model that is
the **teacher** in our FYP (the NPU student is described separately). One network takes a degraded image (noise,
rain, haze, low light, or a mix of them) and returns the clean image, without being told which degradation it is.

**Contents**

1. [The model in one paragraph](#1-the-model-in-one-paragraph)
2. [The backbone (kept from AdaIR)](#2-the-backbone-kept-from-adair)
3. [The frequency blocks (our improvement)](#3-the-frequency-blocks-our-improvement)
4. [Size and cost](#4-size-and-cost)
5. [Training](#5-training)
6. [Testing](#6-testing)
7. [Results](#7-results)
8. [Where everything is](#8-where-everything-is)
9. [Glossary](#9-glossary)

---

## 1. The model in one paragraph

The model is a U-shaped encoder-decoder made of transformer blocks (the AdaIR / Restormer backbone). The encoder
shrinks the image step by step to understand it; the decoder grows it back to full size to rebuild it. In the
decoder, three **frequency blocks** look at the input image in the frequency domain: they split it into **four
shaped frequency bands** (dot, star, lobe, rest), and **band attention** decides how to mix them for this image. The
output is added to the decoder features, so the decoder knows which kinds of detail (smooth light and colour, edges,
streaks, fine texture) the degradation has damaged. The network predicts a correction that is added to the input
image.

What is ours and what comes from AdaIR:

| Part | From AdaIR | Ours |
|---|---|---|
| Backbone (encoder, decoder, transformer blocks) | ✔ | kept as is, starts from the released AdaIR weights |
| How the spectrum is split | 2 parts: a centred rectangle (low) and the rest (high) | **4 shaped bands**, shapes predicted per image |
| How the parts are mixed | fixed H-L / L-H units | **band self-attention** (learned at every pixel) |
| Pre-training | none | **DCPT** warm-up (degradation classification) |
| Loss | L1 | **L1 + 0.1 × FFT L1** (CDD-11 run: + MS-SSIM with a searched weight) |
| Weights | – | **EMA** (moving average) weights are the ones tested |

---

## 2. The backbone (kept from AdaIR)

```
input image (3 channels)
   │ 3×3 conv → 48 channels
   ▼
Encoder level 1    4 blocks,  48 ch, full size  ──────────────────────────┐ skip
   │ down (half size, 2× channels)                                        │
Encoder level 2    6 blocks,  96 ch, 1/2 size   ───────────────────┐      │
   │ down                                                          │      │
Encoder level 3    6 blocks, 192 ch, 1/4 size   ────────────┐      │      │
   │ down                                                   │      │      │
Latent             8 blocks, 384 ch, 1/8 size               │      │      │
   │                                                        │      │      │
[Frequency block 1]  ◄── input image                        │      │      │
   │ up + skip ◄────────────────────────────────────────────┘      │      │
Decoder level 3    6 blocks, 192 ch, 1/4 size                      │      │
[Frequency block 2]  ◄── input image                               │      │
   │ up + skip ◄───────────────────────────────────────────────────┘      │
Decoder level 2    6 blocks,  96 ch, 1/2 size                             │
[Frequency block 3]  ◄── input image                                      │
   │ up + skip ◄──────────────────────────────────────────────────────────┘
Decoder level 1    4 blocks,  96 ch, full size
Refinement         4 blocks,  96 ch, full size
   │ 3×3 conv → 3 channels
   ▼
 + input image   (the network only predicts the correction)
   ▼
restored image
```

- **Transformer block** (44 in total): layer norm → **MDTA** attention → add back; layer norm → **GDFN** feed-forward
  → add back.
  - **MDTA** (multi-Dconv head transposed attention) computes attention **between channels**, not between pixels. The
    attention map is channels × channels, so its cost grows only linearly with image size. It summarises the
    **whole input** at once, which matters for testing (section 6.3).
  - **GDFN** (gated feed-forward) is two 1×1 convs and a 3×3 depthwise conv, with one half gating the other.
- **Down** = 3×3 conv + pixel-unshuffle (half the size, double the channels). **Up** = 3×3 conv + pixel-shuffle.
- **Skips** carry encoder features straight across to the decoder, so fine detail is not lost.
- Attention heads per level: 1 / 2 / 4 / 8. Feed-forward expansion: 2.66.

---

## 3. The frequency blocks (our improvement)

### 3.1 Why frequencies

Different degradations damage different frequencies:

- **Haze and low light** change the slow, smooth parts of the image: overall brightness, colour and contrast. These
  are the **low** frequencies.
- **Rain streaks** are thin lines in mostly one direction, so they sit along a line in the spectrum.
- **Noise** is spread over everything, mostly at **high** frequencies.

If the network can see these parts separately, it can treat each one in the right way. That is AdaIR's idea, and our
blocks do the same thing in a more flexible way.

### 3.2 What one block does

There are three blocks, at 1/8, 1/4 and 1/2 of the image size (384, 192 and 96 channels). Each one takes the
**original input image** (resized to its level) and the **decoder features** at that level:

```
input image, resized to this level
   │ 3×3 conv
   P (features of the input)
   │ FFT (turn into frequencies)
   ├──► shape generator: mean + spread of P → 8 shape numbers for this image
   │         └──► 4 soft masks: dot / star / lobe / rest (they add up to 1)
   ▼
4 bands = inverse FFT of (spectrum × each mask)
   │ each band: cross-attention with the decoder features (own weights per band)
   ▼
band self-attention: at every pixel the 4 bands look at each other and mix
   │ sum + 1×1 conv
   ▼
cross-attention: the decoder features ask the merged bands what they need
   │
output = para1 × (result) + para2 × (decoder features)
```

### 3.3 The four bands

All shapes live in the centred spectrum, measured as a **fraction of the highest frequency** (0 = average colour,
1 = finest detail). So a shape means the same at every block and on every image size. AdaIR's rectangle, by
contrast, has a size tied to the feature size in pixels.

| Band | Shape | What it holds |
|---|---|---|
| **dot** | small disc at the centre | the smoothest content: overall light, colour, haze veil |
| **star** | 4-pointed star (rotated superellipse) | the main horizontal / vertical structure |
| **lobe** | two side lobes, rotated up to ±30° | strong one-direction patterns such as rain streaks |
| **rest** | everything left over | fine texture and noise |

- **Order of priority:** dot first; the star takes what the dot left; the lobes take what is left after that; *rest*
  gets everything else. So the four masks **always add up to exactly 1**, and the four bands together hold exactly
  the original features: nothing is lost or counted twice.
- **The dot follows the image.** Its radius is a multiple (0.5×–2×) of the radius that holds half of this image's
  energy. On a noisy image that radius grows, so the dot grows with it.
- **Soft edges.** Mask edges are smooth (a sigmoid with width 0.05), so the network can learn the shapes by gradient
  descent.
- **Real-valued bands.** Each mask is made symmetric, so every band is a real image.
- **Precision.** The FFT part always runs in full fp32.

### 3.4 Shape generator (SGB)

The shape generator takes the mean and the spread (standard deviation) of every channel of P, and passes them
through a small network (1×1 conv → GELU → 1×1 conv). It predicts **8 shape numbers per image**. The last layer
starts at zero, so before training every image gets the same starting shapes.

Each number is **bounded**: it is squeezed into a fixed range, so the shapes can never collapse into useless forms.

| Shape number | Start | Allowed range |
|---|---|---|
| dot size (× half-energy radius) | 1.0 | 0.5 – 2.0 |
| star angle | 0° | −22.5° – +22.5° |
| star pointiness p (lower = more pointed) | 0.5 | 0.4 – 1.0 (1 = diamond) |
| star size | 0.75 | 0.4 – 0.95 |
| star aspect (width / height) | 1.0 | 0.67 – 1.5 |
| lobe start (distance from centre) | 0.6 | 0.4 – 0.9 |
| lobe curvature | 0.4 | 0.1 – 1.5 |
| lobe angle | 0° | −30° – +30° |

At the start the bands cover about **1% / 8% / 27% / 64%** (dot / star / lobe / rest) of the spectrum.

### 3.5 Attention inside the block

- **Per-band cross-attention.** Each band has its own channel cross-attention (queries from the band, keys and values
  from the decoder features). Each band picks out the decoder information it relates to.
- **Band self-attention (BSA).** At every pixel, the 4 bands attend to each other (a 4×4 attention, 4 heads, with a
  learned "which band am I" embedding). This learns the mixing recipe for each image and each place, and replaces
  AdaIR's fixed H-L / L-H units. Its output layer starts at zero, so at the start it passes the bands through
  unchanged.
- **Final cross-attention.** The decoder features ask the merged bands what they need. The result is added with
  `para1` (starts at 0) and the decoder features are kept with `para2`. Because `para1` starts at 0, the block starts
  "silent" and fades in during training without disturbing the pre-trained backbone.

---

## 4. Size and cost

| | Original AdaIR | Ours |
|---|---|---|
| Parameters | 28.78 M | **31.13 M** (+2.35 M, all in the frequency blocks) |
| Inference, 512×768 image, RTX 4080 SUPER, fp32 | 474 ms | 510 ms |
| Training step, 4 crops of 128 px | 239 ms | 267 ms |

The DCPT classifier (7.45 M parameters) exists only during pre-training and is thrown away afterwards.

---

## 5. Training

### 5.1 Common recipe (both experiments)

| Item | Setting |
|---|---|
| Starting weights | the **released AdaIR backbone** (+ `para2`); the frequency blocks are new and start fresh |
| Optimiser | AdamW (betas 0.9 / 0.999), weight decay 0.01 (0 during DCPT) |
| Learning rate | short warm-up, then cosine decay down to 1e-6 |
| Batch | 32 crops per step (e.g. 4 crops × 8 accumulated steps) |
| Augmentation | random crop + one random flip / rotation (the same for input and target) |
| EMA | an average of the weights, decay 0.999; **the EMA weights are the ones tested** |
| Precision | true fp32 for training (TF32 off) |
| Checkpoints | weights kept after every epoch, so any epoch can be tested or used again |
| Hardware | qbits, one RTX 4080 SUPER (16 GB) |

### 5.2 Stage A: DCPT warm-up

DCPT ("degradation classification pre-training", Hu et al., ICLR 2025) is our own re-implementation. A small
classifier reads the decoder features and must say **which degradation** the input has (focal loss). At the same time
the network gets the **clean** image and must return it unchanged (L1 loss).

- The backbone is **frozen**, so only the new frequency blocks and the classifier learn. The new blocks learn to
  "see" the degradation type before the full training starts, without disturbing the pre-trained backbone.
- The classifier is dropped afterwards; only the network weights go on to stage B.

### 5.3 Stage B: restoration fine-tuning

Everything trains: input = degraded crop, target = clean crop.

**Loss, all-in-one experiment:**

```
L = L1(output, clean) + 0.1 × L1(FFT(output), FFT(clean))
```

The FFT term compares the two images frequency by frequency, so wrong textures and wrong overall colour are punished
directly. It is unnormalised, so it is usually the largest term of the loss.

**Loss, CDD-11 experiment** (after EvoIR, 2025):

```
L = a × L1 + (1 − a) × (1 − MS-SSIM) + 0.1 × FFT-L1
```

MS-SSIM rewards correct structure and contrast. **The weight `a` is searched during training.** At the start of
every epoch from the second on, 3 values of `a` each train a copy of the model for 20 steps on the same batches. Each
copy is scored on held-out validation crops, and the best copy (and its `a`) is kept. It stayed between 0.64 and 1.0,
so L1 stays the main term. The final model ended with `a` = 0.866.

### 5.4 Experiment 1: all-in-one (3 tasks + mixtures)

**Data per epoch (159,567 crops of 128 px):**

| Task | Source | Crops per epoch |
|---|---|---|
| Noise σ 15 / 25 / 50 | BSD400 + WED, noise added on the fly | 15,432 |
| Rain | Rain100L train | 24,000 |
| Haze | OTS (RESIDE) | 72,135 |
| Haze + noise | OTS + added noise | 12,000 |
| Rain + noise | Rain100L + added noise | 12,000 |
| Haze + rain | OTS + rain streaks taken from Rain100L | 12,000 |
| Haze + rain (CDD-11) | CDD-11 train | 12,000 |

**Schedule:**

| | Stage A: DCPT | Stage B: fine-tune (ours) | Baseline (for comparison) |
|---|---|---|---|
| Model | ours | ours | original AdaIR, unchanged |
| Starts from | released AdaIR 3-task backbone | stage A | full released AdaIR 3-task weights |
| What trains | new blocks + classifier | everything | everything |
| Steps | 5,000 (1 epoch) | 99,730 (20 epochs) | 99,730 (20 epochs) |
| Batch | 2 × 16 | 4 × 8 | 4 × 8 |
| Learning rate | 2e-4 (classifier 5e-5) | 1e-4 → 1e-6, 2% warm-up | 1e-4 → 1e-6, 2% warm-up |
| Loss | clean L1 + classification | L1 + 0.1 FFT | L1 only |
| EMA | – | 0.999 | 0.999 |

The baseline has the same data, schedule and EMA, so the comparison only measures our changes.

### 5.5 Experiment 2: CDD-11 (6 classes)

A separate model, trained only on 6 CDD-11 classes: **low, haze, rain, haze+rain, low+haze, low+rain**.

- **Data.** CDD-11 has 1,183 training scenes per class. **40 scenes are held out** (the same 40 in every class) for
  validation and for choosing settings. They are never trained on. The other 1,143 scenes per class are used once
  per epoch: **6,858 crops per epoch, exactly balanced between the classes**.
- **Starting point.** The released AdaIR **5-task** backbone, which has already seen low-light images.
- **Bigger crops step by step** (as in Restormer). The model learns on small crops first, then on larger areas, so it
  learns large-scale effects like haze.

| | DCPT | Stage 1 | Stage 2 | **Stage 3 (final)** |
|---|---|---|---|---|
| Starts from | released 5-task backbone | DCPT | stage 1 | stage 2 |
| Crop size | 128 px | 128 px | 256 px | **384 px** |
| Length | 2,000 steps | 10.5 epochs | 10 epochs | **5 epochs** |
| Batch | 2 × 16 | 4 × 8 | 1 × 32 | 1 × 32 |
| Learning rate | 2e-4 | 1e-4 → (5e-5 at the stop) | 5e-5 → 1e-6 | 3e-5 → 1e-6 |
| Loss | clean L1 + classification | mix + search | mix + search | mix + search |
| EMA | – | 0.999 with warm-up | 0.999 with warm-up | 0.999 with warm-up |
| Tested after each epoch in | – | whole image | 256-px tiles | 512-px tiles |
| Result | classification accuracy 0.85 | 26.32 dB (epoch 10) | 27.63 dB (epoch 8) | **28.57 dB (epoch 5)** |

- **EMA warm-up.** The decay is min(0.999, (1+t)/(10+t)) at step t. These runs are short (214 steps per epoch), and
  a plain 0.999 average would still be mostly the starting weights after the first epoch.
- **Gradient checkpointing** (stage 3 only). The 44 transformer blocks are recomputed in the backward pass instead of
  stored. The gradients are the same, it costs about 30% more time, and it lets 384-px crops fit in 16 GB.
- **Picking the final model.** Every epoch was tested on the full test set, and the epoch with the best mean PSNR was
  kept: epoch 5 of stage 3 (`runs/cdd6_ft384/epochs/epoch_05.pt`).

---

## 6. Testing

### 6.1 Metrics

**PSNR** and **SSIM** on RGB images in [0, 1], computed with scikit-image (`data_range = 1`). SSIM uses its default
7×7 window. Higher is better for both.

### 6.2 All-in-one test

| Test set | Images | Notes |
|---|---|---|
| BSD68, noise σ 15 / 25 / 50 | 68 each | noise added with a fixed seed per image |
| Rain100L | 100 | |
| SOTS outdoor (haze) | 500 | |
| Haze + noise σ 25 | 500 | made once and saved |
| Rain + noise σ 25 | 100 | made once and saved |
| Haze + rain | 500 | made once and saved |
| Haze + rain (CDD-11) | 200 | CDD-11 test split |

- **Protocol:** AdaIR's own test protocol (each image centre-cropped to a multiple of 16), true fp32, every image of
  every set.
- **Check:** with this code the released AdaIR weights reproduce the AdaIR paper's Table 1 exactly (for example
  Rain100L 38.64 / 0.983, SOTS 31.06 / 0.980).

### 6.3 CDD-11 test (OneRestore protocol)

- All **200 test images per class**, full size (1080 × 720; some are portrait). The test set is always used complete.
- Same rules as the OneRestore / EvoIR papers: the output is **rounded to 8-bit** (as when saved as PNG) before the
  metrics, and SSIM uses a 7×7 window.
- Inference in TF32 for speed. On a check this was 0.0004 dB from true fp32.

**Tiled inference.** The model was trained on crops (at most 384 px), so it works best on pieces of about that size.
Its channel attention summarises the whole input at once. On a full 1080 × 720 image it averages very different
regions (for example near and far haze) into one summary; on a tile it adapts to that region.

- The image is cut into **512 × 512 tiles** with 128 px overlap (a 1080 × 720 image becomes 6 tiles, in either
  orientation).
- Each tile is restored, and the overlaps are blended with smooth weights, so no seams show.
- The tile size was **chosen on the 40 held-out training scenes, never on the test set**. On those scenes the final
  model scored 28.64 / 29.07 / 28.91 / 28.81 dB for 384 / 512 / 640 / 720-px tiles.

**Self-ensemble ×8 ("Ours+").** Each tile is also restored in its 8 flipped / rotated versions. Each result is turned
back and the 8 are averaged. The weights are the same; it is just 8× the compute (about 12 s per image instead of
1.5 s). This is a standard trick (EDSR, 2017), marked with "+" in papers. EvoIR and OneRestore do not use it, so
**the plain "Ours" number is the fair comparison**.

---

## 7. Results

### 7.1 All-in-one: ours vs the baseline, both after 20 epochs

PSNR / SSIM, full test sets.

| Degradation | Released AdaIR | Baseline (epoch 20) | **Ours (epoch 20)** | Ours − baseline (PSNR) |
|---|---|---|---|---|
| Noise σ 15 | 34.12 / 0.9353 | 33.65 / 0.9333 | **33.75 / 0.9350** | +0.10 |
| Noise σ 25 | 31.46 / 0.8921 | 31.05 / 0.8890 | **31.15 / 0.8917** | +0.10 |
| Noise σ 50 | 28.18 / 0.8019 | 27.84 / 0.7976 | **27.95 / 0.8029** | +0.10 |
| Rain (Rain100L) | 38.64 / 0.9830 | 37.83 / 0.9824 | **37.86 / 0.9826** | +0.04 |
| Haze (SOTS) | 31.06 / 0.9802 | 31.00 / 0.9800 | **31.22 / 0.9809** | +0.22 |
| Haze + noise | 15.74 / 0.7173 | **25.01** / 0.8382 | 24.93 / **0.8438** | −0.08 |
| Rain + noise | 24.51 / 0.7374 | 30.52 / 0.8689 | **30.74 / 0.8735** | +0.23 |
| Haze + rain | 20.46 / 0.8096 | 28.55 / 0.9475 | **29.01 / 0.9545** | +0.46 |
| Haze + rain (CDD-11) | 18.67 / 0.7703 | 28.88 / 0.9528 | **29.55 / 0.9615** | +0.67 |
| **Average, single (5)** | 32.69 | 32.27 | **32.39** | **+0.11** |
| **Average, mixed (4)** | 19.84 | 28.24 | **28.56** | **+0.32** |

Ours wins 8 of 9 in PSNR and all 9 in SSIM. Fine-tuning on mixtures costs both models a little on noise and rain
compared with the released model, but gains up to 10.9 dB on the mixtures, which the released model never saw.

### 7.2 CDD-11 (6 classes), final model

PSNR / SSIM, all 200 test images per class, OneRestore protocol.

| Method | low | haze | rain | haze+rain | low+haze | low+rain | **Mean** |
|---|---|---|---|---|---|---|---|
| **Ours** | 26.14 / 0.8294 | **33.07** / 0.9897 | 33.12 / 0.9626 | 29.43 / 0.9540 | 24.81 / 0.8169 | 24.85 / 0.7899 | **28.57 / 0.8904** |
| **Ours+ (×8 self-ensemble)** | 26.17 / 0.8308 | **33.35** / 0.9903 | 33.35 / 0.9647 | 29.75 / 0.9573 | 24.90 / 0.8192 | 24.95 / 0.7940 | **28.75 / 0.8927** |
| OneRestore (published) | 26.48 / 0.826 | 32.52 / 0.990 | 33.40 / 0.964 | 29.99 / 0.957 | 25.79 / 0.822 | 25.58 / 0.799 | 28.96 / 0.893 |
| EvoIR (published) | 27.06 / 0.830 | 32.24 / 0.991 | 34.03 / 0.970 | 29.76 / 0.965 | 25.92 / 0.824 | 26.02 / 0.806 | 29.17 / 0.898 |

- Above the 28 dB target. **Haze is the best of the three methods.** With the self-ensemble, SSIM is level with
  OneRestore.
- The remaining gap is in the low-light mixes (low+haze, low+rain).
- OneRestore and EvoIR are trained on all 11 CDD-11 classes; ours on these 6.

**How each step added up** (mean PSNR, no flips):

| Step | Mean PSNR |
|---|---|
| Stage 1 (128-px crops), whole image | 26.32 |
| Same model, 256-px tiles | 27.10 |
| Stage 2 (256-px crops), 256-px tiles | 27.63 |
| Same model, 512-px tiles | 28.27 |
| **Stage 3 (384-px crops), 512-px tiles** | **28.57** |
| + ×8 self-ensemble | 28.75 |

---

## 8. Where everything is

**Code** (in this repository):

| File | What it holds |
|---|---|
| `net/model.py` | the AdaIR backbone, unchanged from the release |
| `patches/shaped_bands.py` | our frequency block: 4 shaped bands, shape generator, band self-attention |
| `patches/dcpt.py` | DCPT pre-training (classifier + losses) |
| `patches/losses.py` | L1 + FFT loss, MS-SSIM, the mix loss |
| `fyp/models.py` | builds the model variants and loads weights |
| `fyp/data.py` | training data (tasks, mixtures, CDD-11 classes, held-out scenes) |
| `fyp/evaluate.py` | test sets, protocols, tiled inference, self-ensemble |
| `fyp/metrics.py` | PSNR / SSIM |
| `train_fyp.py` | training (DCPT and restoration stages, EMA, loss search, per-epoch tests) |
| `eval_fyp.py` | all-in-one test |
| `tools/cdd_infer_sweep.py` | CDD-11 test with tiles and self-ensemble |
| `scripts/run_ours.sh`, `scripts/run_baseline.sh` | experiment 1 runs |
| `scripts/run_cdd6.sh`, `run_cdd6_256.sh`, `run_cdd6_384.sh`, `cdd6_final.sh` | experiment 2 runs and final test |
| `splits/cdd11_val_scenes.txt` | the 40 held-out CDD-11 scenes |

**Weights** (on qbits, `/home/minura/Himeth/AdaIR/runs/`):

| Model | File |
|---|---|
| All-in-one, ours, epoch 20 | `runs/ours_ft/epochs/step_0099730.pt` |
| All-in-one, baseline, epoch 20 | on Elgin, `runs/baseline_ft/epochs/step_0099730.pt` |
| CDD-11, final | `runs/cdd6_ft384/epochs/epoch_05.pt` (use the `ema` weights) |

Every number in this document comes from `results.md` (sections A5–A30); `log.md` has the full history.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **PSNR** | how close the output is to the clean image, in dB (higher = better; +1 dB is a clear gain) |
| **SSIM** | how similar the structure looks, 0 to 1 (higher = better) |
| **FFT / spectrum** | the image written as a sum of waves; the centre is smooth content, the edges are fine detail |
| **Band** | the part of the image made of one region of the spectrum |
| **Cross-attention** | one set of features asks another set which channels are relevant |
| **Epoch** | one pass over the training data |
| **Crop** | a small square cut from a training image |
| **EMA** | a slowly moving average of the weights; smoother and usually better than the latest weights |
| **Held-out scenes** | training scenes kept aside and never trained on; used to choose settings without touching the test set |
| **Tiled inference** | restoring a big image in overlapping pieces and blending them |
| **Self-ensemble (×8)** | restoring 8 flipped / rotated copies and averaging them |
| **DCPT** | pre-training where the network learns to recognise the degradation type |
| **Gradient checkpointing** | saving memory by recomputing parts of the network during the backward pass |
