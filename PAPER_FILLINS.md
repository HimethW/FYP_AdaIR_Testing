# Paper fill-ins for the teacher (BandIR)

These are ready-to-paste LaTeX pieces for the teacher parts of the draft (`Reviewer_copy.pdf`): the to-dos marked in
red, plus a few teacher sentences that are not marked but don't match our runs.

- **Our numbers** come from the saved result files: `results.md` A24 and A30, and the epoch-20 mask shapes.
- **Other methods' numbers** come from their papers:
  - MoCE-IR-S CDD-11: MoCE-IR (CVPR'25) Table 3. The PSNR values were cross-checked in a second paper (DVANet).
  - OneRestore and EvoIR: EvoIR Table 4.
  - PromptIR CDD-11: MoCE-IR Table 3. Its 6-class mean equals the 26.34 already in the draft.
  - Three-task numbers: the draft's own Table 2.
- **Citation keys** such as `\cite{zamfir2025moceir}` are placeholders. Swap in the keys your `.bib` already uses.

---

## 0. Status of every teacher to-do

| Where (draft line) | To-do | Status |
|---|---|---|
| Sec. 5.3, l. 537–539; Table 3 | Rows for MoCE-IR-S, BioIR | **MoCE-IR-S filled** (section 1). **BioIR:** three-task value filled; its CDD-11 numbers are only in its NeurIPS PDF, which I haven't read yet |
| Sec. 5.3, l. 537–539; Table 3 | Row for StaticIR distilled from BandIR | **Not trained.** Two text options in section 4 |
| Intro l. 107–109, contributions l. 125–126, Sec. 4.3 l. 465–467 | StaticIR distilled from BandIR | **Not trained.** Same as above, section 4 |
| Sec. 5.4, l. 540 | Ablations | **Teacher part:** the CDD-11 recipe table can go in now (section 6). The per-component ablation **needs new runs** (plan in section 6) |
| Not marked | Wording that doesn't match our runs | Fixes in sections 2 and 5 |
| Not marked | `[?]` citations in the teacher text | BibTeX in section 7 |

---

## 1. Comparison tables

### 1.1 Table 3, completed (single, composite and CDD-11 averages)

Single = the five three-task sets; composite = our four composite sets; CDD-11 = the mean over our 6 classes.

| Method | Single | Composite | CDD-11 (6 classes) |
|---|---|---|---|
| PromptIR | 32.06 | – | 26.34 |
| OneRestore | – | – | 28.96 |
| EvoIR | – | – | 29.17 |
| **MoCE-IR-S** (new row) | 32.57 | – | **29.44** |
| **BioIR** (new row) | **32.87** | – | *from its paper, still to fill* |
| AdaIR, released | 32.69 | 19.84 | – |
| AdaIR, fine-tuned | 32.27 | 28.24 | – |
| BandIR (ours) | 32.39 | **28.56** | 28.57 |

**MoCE-IR-S is above BandIR on CDD-11** (29.44 vs 28.57, −0.87 dB). The text in section 2 says this plainly and keeps
our real strengths:
- the controlled comparison with AdaIR;
- the best haze result.

```latex
\begin{table}[t]
\centering
\caption{BandIR as a teacher for composite degradations (average PSNR in dB). ``Fine-tuned'' is the original
AdaIR trained with exactly the data, schedule and EMA of BandIR. Single: the five three-task sets; composite: the
four composite sets; CDD-11: the mean over the six classes we use (low light, haze, rain and their three pairs),
computed from each method's per-class results. The CDD-11 result of BandIR comes from a separate model trained on
these six classes; the other methods are trained on all eleven. Results of other methods are taken from their
papers; --: not reported. Best in bold. Per-set results are in Sec.~C.}
\label{tab:bandir}
\small
\begin{tabular}{lccc}
\toprule
Method & Single & Composite & CDD-11 \\
\midrule
PromptIR~\cite{potlapalli2023promptir} & 32.06 & -- & 26.34 \\
OneRestore~\cite{guo2024onerestore} & -- & -- & 28.96 \\
EvoIR~\cite{ma2025evoir} & -- & -- & 29.17 \\
MoCE-IR-S~\cite{zamfir2025moceir} & 32.57 & -- & \textbf{29.44} \\
BioIR~\cite{cui2025bioinspired} & \textbf{32.87} & -- & \todo{?} \\
\midrule
AdaIR, released~\cite{cui2025adair} & 32.69 & 19.84 & -- \\
AdaIR, fine-tuned & 32.27 & 28.24 & -- \\
BandIR (ours) & 32.39 & \textbf{28.56} & 28.57 \\
% StaticIR distilled from BandIR & ... & ... & ...   (add once it is trained, see Sec. 4 of PAPER_FILLINS.md)
\bottomrule
\end{tabular}
\end{table}
```

The "composite" column is "–" for the other methods because nobody reports results on our four composite test sets.
Filling it would mean running their released three-task checkpoints on those sets. They were not trained on
mixtures, so like the released AdaIR (19.84) they would mostly fail. I don't think that is worth doing; say if you
want it.

### 1.2 Table 4 (supplement), per test set, now with SSIM

This makes the claim "higher SSIM on all 9" checkable: right now SSIM appears in no table. Best is in bold and second
underlined, as in the draft.

```latex
\begin{table*}[t]
\centering
\caption{BandIR against AdaIR, per test set (PSNR in dB, SSIM below). ``Fine-tuned'' is the original AdaIR trained
with exactly the data, schedule and EMA of BandIR. Composite sets: haze+noise (H+N), rain+noise (R+N), synthetic
haze+rain (H+R) and the haze+rain class of CDD-11; noise at $\sigma=25$. SOTS uses all 500 images. Best in bold,
second underlined.}
\label{tab:bandir_sets}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l cccccc ccccc}
\toprule
& \multicolumn{6}{c}{Single degradations} & \multicolumn{5}{c}{Composite degradations} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-12}
Model & $\sigma$=15 & $\sigma$=25 & $\sigma$=50 & Rain100L & SOTS & Avg. & H+N & R+N & H+R & CDD H+R & Avg. \\
\midrule
AdaIR, released~\cite{cui2025adair}
  & \textbf{34.12} & \textbf{31.46} & \textbf{28.18} & \textbf{38.64} & \underline{31.06} & \textbf{32.69}
  & 15.74 & 24.51 & 20.46 & 18.67 & 19.84 \\
  & \textbf{.9353} & \textbf{.8921} & \underline{.8019} & \textbf{.9830} & \underline{.9802} & \underline{.9185}
  & .7173 & .7374 & .8096 & .7703 & .7587 \\
AdaIR, fine-tuned
  & 33.65 & 31.05 & 27.84 & 37.83 & 31.00 & 32.27
  & \textbf{25.01} & \underline{30.52} & \underline{28.55} & \underline{28.88} & \underline{28.24} \\
  & .9333 & .8890 & .7976 & .9824 & .9800 & .9165
  & \underline{.8382} & \underline{.8689} & \underline{.9475} & \underline{.9528} & \underline{.9019} \\
BandIR (ours)
  & \underline{33.75} & \underline{31.15} & \underline{27.95} & \underline{37.86} & \textbf{31.22} & \underline{32.39}
  & \underline{24.93} & \textbf{30.74} & \textbf{29.01} & \textbf{29.55} & \textbf{28.56} \\
  & \underline{.9350} & \underline{.8917} & \textbf{.8029} & \underline{.9826} & \textbf{.9809} & \textbf{.9186}
  & \textbf{.8438} & \textbf{.8735} & \textbf{.9545} & \textbf{.9615} & \textbf{.9083} \\
\bottomrule
\end{tabular}
\end{table*}
```

**SOTS on the 424 unseen images** (Sec. 5.1 promises these; optional sentence for Sec. C):

> On the 424 SOTS images that do not appear in OTS, BandIR reaches 30.21\,dB, the fine-tuned AdaIR 30.00\,dB and the
> released AdaIR 30.07\,dB.

### 1.3 Table 5 (supplement), CDD-11 per class, with the new rows

Best in bold, PSNR and SSIM separately. The BioIR row may change which values are bold.

```latex
\begin{table*}[t]
\centering
\caption{Six classes of CDD-11, per class (PSNR / SSIM, all 200 test images per class). The mean is over the six
classes. Competing results are taken from their papers; those methods are trained on all eleven classes, BandIR on
these six. BandIR is evaluated in $512\times512$ tiles, a size chosen on held-out training scenes (Sec.~C). Best in
bold.}
\label{tab:cdd}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccccc}
\toprule
Method & Low & Haze & Rain & Haze+rain & Low+haze & Low+rain & Mean \\
\midrule
PromptIR~\cite{potlapalli2023promptir} & 26.32 / .805 & 26.10 / .969 & 31.56 / .946 & 24.54 / .924 & 24.49 / .789 & 25.05 / .771 & 26.34 / .867 \\
OneRestore~\cite{guo2024onerestore} & 26.48 / .826 & 32.52 / .990 & 33.40 / .964 & \textbf{29.99} / .957 & 25.79 / .822 & 25.58 / .799 & 28.96 / .893 \\
EvoIR~\cite{ma2025evoir} & 27.06 / \textbf{.830} & 32.24 / \textbf{.991} & 34.03 / \textbf{.970} & 29.76 / \textbf{.965} & 25.92 / \textbf{.824} & 26.02 / \textbf{.806} & 29.17 / \textbf{.898} \\
MoCE-IR-S~\cite{zamfir2025moceir} & \textbf{27.26} / .824 & 32.66 / .990 & \textbf{34.31} / \textbf{.970} & 29.93 / .964 & \textbf{26.24} / .817 & \textbf{26.25} / .800 & \textbf{29.44} / .894 \\
BioIR~\cite{cui2025bioinspired} & \todo{?} & \todo{?} & \todo{?} & \todo{?} & \todo{?} & \todo{?} & \todo{?} \\
BandIR (ours) & 26.14 / .829 & \textbf{33.07} / .990 & 33.12 / .963 & 29.43 / .954 & 24.81 / .817 & 24.85 / .790 & 28.57 / .890 \\
\bottomrule
\end{tabular}
\end{table*}
```

The 6-class means were computed from the papers' per-class values:
- MoCE-IR-S 29.44 / .894;
- PromptIR 26.34 / .867, the same as the 26.34 the draft already uses.

---

## 2. Sec. 5.3: replacement paragraph (fills the to-do and fixes three issues)

What it changes, and why:

1. **"To isolate the frequency blocks" is not accurate.** The fine-tuned AdaIR used **L1 only, with no DCPT and no
   frequency loss** (results.md A19). The comparison therefore measures the bands, BSA, DCPT and the frequency loss
   together. The new wording says what was matched.
2. **The new rows are added**, and the CDD-11 gap is stated honestly.
3. **The single-task drop is explained.** BandIR's single-task average (32.39) is below the released AdaIR's (32.69).
   A reviewer will ask why.

```latex
Table~\ref{tab:bandir} evaluates BandIR. To separate our changes from the effect of the training data, we
fine-tune the original AdaIR with exactly the data, schedule and EMA of BandIR; BandIR additionally uses the DCPT
warm-up and the frequency-domain loss. Under this matched training, BandIR is better by 0.11\,dB on the
single-degradation sets and by 0.32\,dB on the composite sets, with higher PSNR on 8 of the 9 sets and higher SSIM
on all 9 (Tab.~\ref{tab:bandir_sets}); the largest gains are on haze with rain (up to 0.67\,dB). Both fine-tuned
models trade some single-degradation accuracy for the composite data, 0.30\,dB for BandIR and 0.42\,dB for AdaIR
relative to the released model, which is trained on single degradations only and fails on composite ones
(19.84\,dB). On the six CDD-11 classes, BandIR reaches 28.57\,dB and is the best method on the haze class
(33.07\,dB); it trails OneRestore by 0.39\,dB and MoCE-IR-S by 0.87\,dB, mostly on the classes with low light
(Tab.~\ref{tab:cdd}).
```

If the BioIR row comes in, add "and BioIR by X dB" (or "and is above BioIR") to the last sentence.

---

## 3. Notes for whoever distils StaticIR from BandIR

- **Weights:**
  - three tasks + composites: `/home/minura/Himeth/AdaIR/runs/ours_ft/epochs/step_0099730.pt`;
  - CDD-11: `runs/cdd6_ft384/epochs/epoch_05.pt`.

  Use the `ema` weights.
- **Same feature taps.** BandIR keeps AdaIR's backbone, so the latent z (1/8, 384 channels) and the first two encoder
  levels have the same sizes as in T_aio. The student's adapters, including A_b (1×1 conv + 2× upsampling from the
  1/16 bottleneck), work unchanged.
- **The haze specialist must be decided.** There is no BandIR dehazing specialist. Either keep AdaIR's T_haze for
  hazy samples (one change: T_aio → BandIR) or use BandIR for every sample. The first changes less, so the comparison
  with the AdaIR-taught student is cleaner.
- **Cost:** BandIR's forward pass is about 8% slower than AdaIR's (510 vs 474 ms on a 512×768 image).

---

## 4. "StaticIR distilled from BandIR" (4 to-dos): two options

The student has not been distilled from BandIR. Either drop the claim now, or fill it in after the run.

### Option A: drop the claim (paper is complete without the run)

- **Intro, l. 099–109.** Change the first sentence and delete the to-do:
  ```latex
  Beyond the student, we also improve the teacher. BandIR builds on the frequency analysis of AdaIR~\cite{cui2025adair}, ...
  ```
  Keep the rest of the paragraph up to "...by 0.32\,dB on average on composite degradations." and delete the to-do
  after it.
- **Contributions, l. 122–126.** End the bullet at "...which improves over AdaIR trained identically, most clearly on
  composite degradations." and delete the to-do.
- **Sec. 4.3, l. 465–467.** Delete the to-do. The paragraph ends at "...and one on CDD-11 (Sec.~C)."
- **Table 3.** No StaticIR row.
- **Abstract:** no change needed. It doesn't claim the distillation.

### Option B: keep the claim (after the student run)

- **Intro:**
  ```latex
  Distilled from BandIR with the same recipe, StaticIR reaches \todo{X}\,dB on the three tasks and \todo{Y}\,dB on the
  composite sets, \todo{Z}\,dB above the student distilled from AdaIR.
  ```
- **Contributions:**
  ```latex
  ..., most clearly on composite degradations, and that the same recipe distills into StaticIR (\todo{+Z}\,dB).
  ```
- **Sec. 4.3:**
  ```latex
  BandIR keeps the backbone of AdaIR, so StaticIR is distilled from it with the recipe of Sec.~4.2 and the same
  feature levels.
  ```
- **Table 3:** add the row "StaticIR from BandIR & X & Y & –".

---

## 5. Other teacher wording that doesn't match our runs

1. **Sec. 4.3, l. 452–453.** The draft says "The new blocks are initialized to the identity, so training starts from
   the released AdaIR." This contradicts Sec. C ("each block returns β ⊙ Y"). Our check (results.md A13) shows the
   start is the released AdaIR **with its frequency branch switched off**, not the full released model. Replace with:
   ```latex
   The new blocks start with $\alpha = 0$, so at initialization the network is the released AdaIR with its
   frequency branch switched off, and training does not disturb the pre-trained backbone.
   ```
2. **Sec. 5.1, l. 478–487.** The draft says "We report PSNR" but Tables 4 and 5 report SSIM too. It also says
   "Accuracies are measured in FP32", but CDD-11 was run in TF32. Replace with:
   ```latex
   We report PSNR, and SSIM where noted, on RGB images, ...
   Accuracies are measured in FP32 (TF32 for CDD-11, which changes the mean PSNR by 0.0004\,dB); ...
   ```
3. **Sec. C, l. 873–875.** The 36 h for the CDD-11 model is wall-clock time and includes a full test after every
   epoch. Replace with:
   ```latex
   The two models take about 73 and 36 hours on one NVIDIA RTX 4080 SUPER, the latter including a full CDD-11
   evaluation after every epoch.
   ```
4. **Sec. C, l. 872–873, "173 GMACs".** This is only in our old LaTeX comment, not in `results.md`. Check that it was
   counted the same way as Table 1 (PyTorch FLOP counter, FFT not counted) and mark it with the same †.

---

## 6. Sec. 5.4 ablations (teacher part)

### 6.1 What can be written now: the CDD-11 recipe

These numbers are measured. They show what the progressive crops and tiled testing add.

```latex
\begin{table}[t]
\centering
\caption{CDD-11 recipe of BandIR (six-class mean, PSNR / SSIM, full test set). Larger training crops and testing in
tiles both help; the tile size is chosen on held-out training scenes. $\times8$: geometric self-ensemble.}
\label{tab:abl_cdd}
\small
\begin{tabular}{llc}
\toprule
Training crops & Test input & Mean \\
\midrule
$128\times128$ & whole image & 26.32 / .874 \\
$128\times128$ & $256$ tiles & 27.10 / .884 \\
$256\times256$ & $256$ tiles & 27.63 / .890 \\
$256\times256$ & $512$ tiles & 28.27 / .888 \\
$384\times384$ & $512$ tiles & \textbf{28.57} / .890 \\
\midrule
$384\times384$ & $512$ tiles, $\times8$ & 28.75 / .893 \\
\bottomrule
\end{tabular}
\end{table}
```

Text:

```latex
On CDD-11, training on larger crops and testing in tiles of about the training size are both needed: the channel
attention of the backbone summarizes its whole input, and on a full $1080\times720$ image it averages regions with
very different haze into one statistic. Testing the $128$-pixel model in $256$-pixel tiles adds 0.78\,dB, and
training on $256$- and then $384$-pixel crops, tested in correspondingly larger tiles, adds another 1.47\,dB
(Tab.~\ref{tab:abl_cdd}).
```

How the numbers add up:
- 27.10 − 26.32 = 0.78;
- 28.57 − 27.10 = 1.47.

The 128-px row is stage 1 at epoch 10, and the 256-px rows are stage 2 at epoch 8.

### 6.2 What is missing: the per-component ablation (needs runs; not started)

There is no experiment that changes one part at a time. A reviewer will ask what the shaped bands add over AdaIR's
rectangle once DCPT and the frequency loss are the same. Our 10-epoch check makes this question sharper: switching
the new blocks off at test time changed PSNR by only −0.22 to +0.05 dB.

Proposed plan, cheap version. Every variant uses the CDD-11 stage-1 schedule (128-px crops, 10 epochs), tested once at
the end:

| Variant | What it isolates | New run? |
|---|---|---|
| BandIR (full) | – | no: stage 1 already exists (26.32 whole image, 27.10 in 256 tiles) |
| AdaIR rectangle + DCPT + frequency loss + mix loss | the 4 shaped bands + BSA | yes, about 5–6 h |
| BandIR without BSA | band self-attention | yes, about 5–6 h (needs a small code switch) |
| BandIR without DCPT | DCPT warm-up | yes, about 3 h |
| BandIR, blocks switched off at test time (final weights, no training) | how much the output uses the new blocks | test only, about 30 min |

That is about 15 h on qbits in total, so it fits in one day. The ablation table would be:

| Variant | CDD-11 mean |
|---|---|
| AdaIR rectangle | ? |
| − BSA | ? |
| − DCPT | ? |
| BandIR | 27.10 |

**Not started. It needs your OK.**

---

## 7. Missing citations in the teacher and setup text (BibTeX)

| Draft location | What `[?]` should cite | Key below |
|---|---|---|
| Sec. 4.3 l. 462: "an ℓ1 loss in the pixel and frequency domains [?]" | frequency-domain loss of MIMO-UNet | `cho2021rethinking` |
| Sec. C l. 850: "with a focal loss [?]" | focal loss | `lin2017focal` |
| Sec. C l. 868: "1−MS-SSIM" (no citation) | MS-SSIM | `wang2003multiscale` |
| Sec. 5.1 l. 471–472: SOTS-outdoor [?], OTS [?] | RESIDE | `li2019benchmarking` |
| Sec. 5.1 l. 472: Rain100L [?] | Rain100L | `yang2017deep` |
| Sec. 5.1 l. 473, 501: BSD68 [?], BSD400 [?] | Berkeley segmentation dataset | `martin2001database` |
| Sec. 5.1 l. 501: WED [?] | Waterloo Exploration Database | `ma2017waterloo` |
| Sec. 5.1 l. 491: AdamW [?] | AdamW | `loshchilov2019decoupled` |
| BioIR [?] (several places) | BioIR | `cui2025bioinspired` |

```bibtex
@inproceedings{cho2021rethinking,
  title     = {Rethinking Coarse-to-Fine Approach in Single Image Deblurring},
  author    = {Cho, Sung-Jin and Ji, Seo-Won and Hong, Jun-Pyo and Jung, Seung-Won and Ko, Sung-Jea},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {4641--4650},
  year      = {2021}
}

@inproceedings{lin2017focal,
  title     = {Focal Loss for Dense Object Detection},
  author    = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {2980--2988},
  year      = {2017}
}

@inproceedings{wang2003multiscale,
  title     = {Multiscale Structural Similarity for Image Quality Assessment},
  author    = {Wang, Zhou and Simoncelli, Eero P. and Bovik, Alan C.},
  booktitle = {The Thirty-Seventh Asilomar Conference on Signals, Systems \& Computers},
  volume    = {2},
  pages     = {1398--1402},
  year      = {2003}
}

@article{li2019benchmarking,
  title   = {Benchmarking Single-Image Dehazing and Beyond},
  author  = {Li, Boyi and Ren, Wenqi and Fu, Dengpan and Tao, Dacheng and Feng, Dan and Zeng, Wenjun and Wang, Zhangyang},
  journal = {IEEE Transactions on Image Processing},
  volume  = {28},
  number  = {1},
  pages   = {492--505},
  year    = {2019}
}

@inproceedings{yang2017deep,
  title     = {Deep Joint Rain Detection and Removal from a Single Image},
  author    = {Yang, Wenhan and Tan, Robby T. and Feng, Jiashi and Liu, Jiaying and Guo, Zongming and Yan, Shuicheng},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1357--1366},
  year      = {2017}
}

@inproceedings{martin2001database,
  title     = {A Database of Human Segmented Natural Images and its Application to Evaluating Segmentation
               Algorithms and Measuring Ecological Statistics},
  author    = {Martin, David and Fowlkes, Charless and Tal, Doron and Malik, Jitendra},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  volume    = {2},
  pages     = {416--423},
  year      = {2001}
}

@article{ma2017waterloo,
  title   = {Waterloo Exploration Database: New Challenges for Image Quality Assessment Models},
  author  = {Ma, Kede and Duanmu, Zhengfang and Wu, Qingbo and Wang, Zhou and Yong, Hongwei and Li, Hongliang
             and Zhang, Lei},
  journal = {IEEE Transactions on Image Processing},
  volume  = {26},
  number  = {2},
  pages   = {1004--1016},
  year    = {2017}
}

@inproceedings{loshchilov2019decoupled,
  title     = {Decoupled Weight Decay Regularization},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}

@inproceedings{cui2025bioinspired,
  title     = {Bio-Inspired Image Restoration},
  author    = {Cui, Yuning and Ren, Wenqi and Knoll, Alois},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

---

## 8. Red to-dos that are not about the teacher

These are for the student / NPU side. I have no data for them.

**Intro and Sec. 3:**
- the three-task number "[TODO: 32.65]" in the intro, l. 096;
- Table 1: HOGformer and DFPIR W8A16 entries ("?");
- Table 1: NAFNet-w32 S26 W8A8 ("?");
- F2: "name the dominant layers", l. 271.

**Experiments:**
- Sec. 5.1: "describe the 300 test crops", l. 488;
- Sec. 5.1: "deraining training data" of the student, l. 502;
- Table 2: InstructIR NPU latency "n.t.";
- Table 2: NAFNet-w32 NPU latency ("?");
- Sec. 5.4: student ablations;
- the Conclusion.

**Supplement A:**
- calibration "confirm for the S26 runs", l. 751;
- "AdaIR 184 s on a Galaxy S25: verify; input size", l. 777.

**Other `[?]` citations:**
- DFPIR, HOGformer, StarIR, IDR, ONNX, Qualcomm AI Hub, DSwinIR, TUR;
- strip pooling and squeeze-and-excitation;
- the "perturbing features" method in the intro, l. 036;
- the second reference in "[35?]" and "[20?]".
