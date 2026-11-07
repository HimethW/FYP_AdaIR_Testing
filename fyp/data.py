"""Training data: AdaIR's three tasks plus mixed degradations, generated per sample.

Each sample is a random patch pair (degraded, clean) from one task, augmented with AdaIR's
random_augmentation (one of 7 flips / rotations), with the task's DCPT class label.

Tasks (label used by DCPT; both haze+rain sources share a class):
  denoise          0  BSD400 + WED clean + N(0, sigma^2), sigma in {15, 25, 50}   (data/Train/Denoise)
  derain           1  Rain100L train pairs                                         (data/Train/Derain)
  dehaze           2  OTS hazy -> clear, the 72,135 in data_dir/hazy/hazy_outside.txt (data/Train/Dehaze)
  haze_noise       3  OTS hazy + N(0, sigma^2) -> clear
  rain_noise       4  Rain100L rainy + N(0, sigma^2) -> clean
  haze_rain        5  OTS hazy + streak layer of a random Rain100L train pair -> clear
  haze_rain_cdd11  5  CDD-11 train haze_rain -> clear                              (data/CDD-11/train)

CDD-11 run (its own 6 DCPT classes, not mixed with the tasks above):
  cdd_low 0, cdd_haze 1, cdd_rain 2, cdd_haze_rain 3, cdd_low_haze 4, cdd_low_rain 5
  CDD-11 train/<class> -> train/clear, minus the 40 scenes in splits/cdd11_val_scenes.txt (held out for
  validation and the loss-mix search; every class shares the same scenes, so no held-out scene is ever trained on)

An epoch has a fixed number of samples per task (`counts`): every item of a task is used
count // n times plus a random remainder, like AdaIR's list repetition. Presets:
  aio3      AdaIR's own all-in-one 3-task composition: denoise 15,432 (5,144 x 3 sigmas),
            derain 24,000 (200 x 120), dehaze 72,135
  aio3_mix  aio3 + 12,000 each of haze_noise, rain_noise, haze_rain, haze_rain_cdd11
            (mixture sizes are our choice; the old ratios are unknown)
  cdd6      the 6 CDD-11 classes, 1,143 each (every training scene once per class per epoch, so the
            classes are exactly balanced): 6,858 samples
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tools.mixtures import add_noise, add_streaks, rain_streaks
from utils.image_utils import random_augmentation

LABELS = {"denoise": 0, "derain": 1, "dehaze": 2, "haze_noise": 3, "rain_noise": 4,
          "haze_rain": 5, "haze_rain_cdd11": 5}
CLASS_NAMES = ["noise", "rain", "haze", "haze+noise", "rain+noise", "haze+rain"]
CDD_CLASSES = ("low", "haze", "rain", "haze_rain", "low_haze", "low_rain")  # CDD-11 folder names
CDD_CLASS_NAMES = ["low", "haze", "rain", "haze+rain", "low+haze", "low+rain"]
for _i, _c in enumerate(CDD_CLASSES):
    LABELS[f"cdd_{_c}"] = _i
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDD_VAL_SPLIT = os.path.join(REPO, "splits/cdd11_val_scenes.txt")
SIGMAS = (15, 25, 50)
PRESETS = {
    "aio3": {"denoise": 15432, "derain": 24000, "dehaze": 72135},
    "aio3_mix": {"denoise": 15432, "derain": 24000, "dehaze": 72135, "haze_noise": 12000,
                 "rain_noise": 12000, "haze_rain": 12000, "haze_rain_cdd11": 12000},
    "cdd6": {f"cdd_{c}": 1143 for c in CDD_CLASSES},
}


def is_cdd(counts):
    return any(t.startswith("cdd_") for t in counts)


def class_names(counts):
    """DCPT class names for a task mix (the CDD-11 run has its own 6 classes)."""
    if is_cdd(counts):
        assert all(t.startswith("cdd_") for t in counts), "CDD-11 classes cannot be mixed with the other tasks"
        return CDD_CLASS_NAMES
    return CLASS_NAMES


def cdd_val_scenes():
    return {l.strip() for l in open(CDD_VAL_SPLIT) if l.strip()}


def parse_tasks(spec):
    """'aio3' / 'aio3_mix' or 'denoise:15432,dehaze:72135,...'."""
    if spec in PRESETS:
        return dict(PRESETS[spec])
    out = {}
    for part in spec.split(","):
        name, n = part.split(":")
        if name not in LABELS:
            raise ValueError(f"unknown task {name}")
        out[name] = int(n)
    return out


def read(path):
    return np.array(Image.open(path).convert("RGB"))


class TrainSet(Dataset):
    def __init__(self, counts, root="data", data_file_dir="data_dir", patch=128, seed=0):
        self.counts, self.root, self.patch = counts, root, patch
        tr = os.path.join(root, "Train")
        self.items = {}
        if "denoise" in counts:
            names = {l.strip() for l in open(os.path.join(data_file_dir, "noisy/denoise.txt"))}
            files = sorted(f for f in os.listdir(f"{tr}/Denoise") if f in names)
            self.items["denoise"] = [f"{tr}/Denoise/{f}" for f in files]
        if counts.keys() & {"derain", "rain_noise", "haze_rain"}:
            lines = [l.strip() for l in open(os.path.join(data_file_dir, "rainy/rainTrain.txt")) if l.strip()]
            self.rain_pairs = [(f"{tr}/Derain/{l}", f"{tr}/Derain/gt/{os.path.basename(l).replace('rain-', 'norain-')}")
                               for l in lines]
            self.items["derain"] = self.items["rain_noise"] = self.rain_pairs
        if counts.keys() & {"dehaze", "haze_noise", "haze_rain"}:
            lines = [l.strip() for l in open(os.path.join(data_file_dir, "hazy/hazy_outside.txt")) if l.strip()]
            pairs = []
            for l in lines:
                name = os.path.basename(l)
                pairs.append((f"{tr}/Dehaze/{l}", f"{tr}/Dehaze/original/{name.split('_')[0]}{os.path.splitext(name)[1]}"))
            self.items["dehaze"] = self.items["haze_noise"] = self.items["haze_rain"] = pairs
        if "haze_rain_cdd11" in counts:
            d = os.path.join(root, "CDD-11/train")
            self.items["haze_rain_cdd11"] = [(f"{d}/haze_rain/{f}", f"{d}/clear/{f}")
                                             for f in sorted(os.listdir(f"{d}/haze_rain"))]
        cdd = [t for t in counts if t.startswith("cdd_")]
        if cdd:
            d = os.path.join(root, "CDD-11/train")
            held_out = cdd_val_scenes()
            for task in cdd:
                c = task[len("cdd_"):]
                self.items[task] = [(f"{d}/{c}/{f}", f"{d}/clear/{f}")
                                    for f in sorted(os.listdir(f"{d}/{c}")) if f not in held_out]
        for task in counts:
            if not self.items.get(task):
                raise RuntimeError(f"no training data found for task {task} under {root}")
        self.epoch_seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Rebuild the (task, item) index for an epoch (repeats + random remainder)."""
        rng = random.Random(self.epoch_seed * 100003 + epoch)
        index = []
        for task, n in self.counts.items():
            m = len(self.items[task])
            idx = list(range(m)) * (n // m) + rng.sample(range(m), n % m)
            index += [(task, i) for i in idx]
        self.index = index

    def __len__(self):
        return len(self.index)

    def _crop(self, *imgs):
        h, w = imgs[0].shape[:2]
        p = self.patch
        if h < p or w < p:  # pad small images (reflect) so a full patch exists
            ph, pw = max(0, p - h), max(0, p - w)
            imgs = [np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="reflect") for im in imgs]
            h, w = imgs[0].shape[:2]
        y, x = random.randint(0, h - p), random.randint(0, w - p)
        return [im[y:y + p, x:x + p] for im in imgs]

    def __getitem__(self, i):
        task, k = self.index[i]
        rng = np.random.default_rng(random.getrandbits(63))
        if task == "denoise":
            (gt,) = self._crop(read(self.items[task][k]))
            lq = add_noise(gt, random.choice(SIGMAS), rng)
        elif task in ("derain", "rain_noise", "dehaze", "haze_noise", "haze_rain", "haze_rain_cdd11") \
                or task.startswith("cdd_"):
            lq_path, gt_path = self.items[task][k]
            lq, gt = self._crop(read(lq_path), read(gt_path))
            if task in ("haze_noise", "rain_noise"):
                lq = add_noise(lq, random.choice(SIGMAS), rng)
            elif task == "haze_rain":
                rp, cp = self.rain_pairs[random.randrange(len(self.rain_pairs))]
                lq = add_streaks(lq, rain_streaks(read(rp), read(cp)), rng)
        lq, gt = random_augmentation(lq, gt)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a).transpose(2, 0, 1)).float().div(255)  # noqa: E731
        return {"lq": to_t(lq), "gt": to_t(gt), "label": LABELS[task], "task": task}


def worker_init(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
