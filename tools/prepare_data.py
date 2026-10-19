"""Download and lay out every dataset for the AdaIR FYP rebuild, in AdaIR's expected layout.

Resumable: each step checks its output and skips if already complete. Downloads go to
data/_downloads/<step>/, are extracted to data/_extract/<step>/, moved into place, then deleted.
data/ is gitignored; the Google Drive backup is the second copy.

Layout produced (AdaIR defaults, options.py / test.py):
  data/Train/Denoise/                BSD400 (as a<id>.jpg, the names data_dir/noisy/denoise.txt expects) + WED (*.bmp)
  data/Train/Derain/{rainy,gt}/      Rain100L train (rain-N.png / norain-N.png), 200 pairs
  data/Train/Dehaze/original/        RESIDE-beta OTS clear (<id>.jpg, 2,061)
  data/Train/Dehaze/synthetic/part1-4/  OTS hazy (72,135, = data_dir/hazy/hazy_outside.txt)
  data/test/denoise/{bsd68,urban100,kodak24}/
  data/test/derain/Rain100L/{input,target}/   rain-NNN.png in both (loader needs identical names)
  data/test/dehaze/{input,target}/            SOTS outdoor (500 hazy, 492 gt as <id>.png)
  data/test/dehaze_ohaze/{input,target}/      O-HAZE, 45 pairs (usual test split 41-45)
  data/test/dehaze_nhhaze/{input,target}/     NH-HAZE, 55 pairs (usual test split 51-55)
  data/CDD-11/{train,test}/{clear,haze_rain}/ CDD-11 haze+rain subset only (1,183 train / 200 test pairs)

Sources (checked 2026-10-19; see log.md): Google Drive IDs from AdaIR's INSTALL.md (Hugging Face
mirror laoduanaaa/Image_Restoration_Datasets as fallback for BSD400 / WED), RESIDE on Box (official),
CBSD68 on GitHub, O-HAZE / NH-HAZE from ETH, CDD-11 from its official Hugging Face repo gy65896/CDD-11
(only the haze_rain + clear folders are read, over HTTP Range, instead of the 26 GB zips).

Usage: python tools/prepare_data.py [--steps bsd400 wed ...] [--root data]
"""
import argparse
import glob
import io
import os
import shutil
import subprocess
import sys
import time
import zipfile

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_MIRROR = "https://huggingface.co/datasets/laoduanaaa/Image_Restoration_Datasets/resolve/main/"
BOX = "https://utexas.app.box.com/index.php?rm=box_download_shared_file&shared_name={share}&file_id=f_{fid}"
OTS_SHARE = "25idwrsn890w03grdr6pls28cy38r91i"
SOTS_SHARE = "uqvnbfo68kns1210z5k5j17cvazavcd1"
CDD = "https://huggingface.co/datasets/gy65896/CDD-11/resolve/main/{split}.zip"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count(path, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif")):
    return sum(1 for f in os.listdir(path) if f.lower().endswith(exts)) if os.path.isdir(path) else 0


def curl(url, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(5):
        r = subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-sS", "-o", out, url])
        if r.returncode == 0:
            return out
        log(f"curl failed ({r.returncode}), retry {attempt + 1}")
        time.sleep(10)
    raise RuntimeError(f"download failed: {url}")


def gdrive(file_id, out, mirror=None):
    import gdown
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(4):
        try:
            if gdown.download(id=file_id, output=out, quiet=True, resume=True):
                return out
        except Exception as e:  # quota exceeded etc.
            log(f"gdown {file_id}: {str(e).splitlines()[0][:120]}")
        time.sleep(30)
    if mirror:
        log(f"Drive failed, using mirror {mirror}")
        return curl(mirror, out)
    raise RuntimeError(f"Drive download failed: {file_id}")


def unzip(zip_path, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    subprocess.run(["unzip", "-q", zip_path, "-d", dest], check=True)
    return dest


def move_all(files, dest, rename=lambda n: n):
    os.makedirs(dest, exist_ok=True)
    for f in files:
        shutil.move(f, os.path.join(dest, rename(os.path.basename(f))))


def find_dir(root, name):
    """First directory called `name` below root (root itself excluded)."""
    for d, subdirs, _ in os.walk(root):
        if d != root and os.path.basename(d) == name:
            return d
    raise FileNotFoundError(f"{name} not in {root}")


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file over HTTP Range requests (to read parts of a zip)."""

    def __init__(self, url):
        self.s = requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, stream=True, timeout=60)
        if r.status_code != 206:
            raise RuntimeError(f"no range support: {r.status_code}")
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Range"].split("/")[-1]), 0
        r.close()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=120)
                data = r.content
                break
            except requests.RequestException:
                time.sleep(5)
        else:
            raise RuntimeError("range read failed")
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


# ---------------------------------------------------------------- steps
def step_bsd400(root):
    dest = os.path.join(root, "Train/Denoise")
    if len(glob.glob(os.path.join(dest, "a*.jpg"))) == 400:
        return log("bsd400: done")
    z = gdrive("1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N", f"{root}/_downloads/bsd400/BSD400.zip",
               HF_MIRROR + "Denoising/Gaussian/BSD400.zip")
    ex = unzip(z, f"{root}/_extract/bsd400")
    # denoise.txt lists BSD400 images as a<id>.jpg; the zip has <id>.jpg (AdaIR issue #4)
    move_all(glob.glob(os.path.join(find_dir(ex, "BSD400"), "*.jpg")), dest, lambda n: "a" + n)
    shutil.rmtree(ex), os.remove(z)
    log(f"bsd400: {len(glob.glob(os.path.join(dest, 'a*.jpg')))} images")


def step_wed(root):
    dest = os.path.join(root, "Train/Denoise")
    if len(glob.glob(os.path.join(dest, "*.bmp"))) == 4744:
        return log("wed: done")
    z = gdrive("1e62XGdi5c6IbvkZ70LFq0KLRhFvih7US", f"{root}/_downloads/wed/WaterlooED.zip",
               HF_MIRROR + "Denoising/Gaussian/WaterlooED.zip")
    ex = unzip(z, f"{root}/_extract/wed")
    move_all(glob.glob(os.path.join(find_dir(ex, "WaterlooED"), "*.bmp")), dest)
    shutil.rmtree(ex), os.remove(z)
    log(f"wed: {len(glob.glob(os.path.join(dest, '*.bmp')))} images")


def step_rain_train(root):
    dest = os.path.join(root, "Train/Derain")
    if count(f"{dest}/rainy") == 200 and count(f"{dest}/gt") == 200:
        return log("rain_train: done")
    z = gdrive("1tiCYr_-ptzwYlRaQOtsI-7seXkrNo-bs", f"{root}/_downloads/rain_train/RainTrainL.zip")
    ex = unzip(z, f"{root}/_extract/rain_train")
    src = find_dir(ex, "RainTrainL")
    move_all(glob.glob(os.path.join(src, "rain-*.png")), f"{dest}/rainy")
    move_all(glob.glob(os.path.join(src, "norain-*.png")), f"{dest}/gt")
    shutil.rmtree(ex), os.remove(z)
    log(f"rain_train: {count(dest + '/rainy')} rainy, {count(dest + '/gt')} gt")


def step_rain_test(root):
    dest = os.path.join(root, "test/derain/Rain100L")
    if count(f"{dest}/input") == 100 and count(f"{dest}/target") == 100:
        return log("rain_test: done")
    z = gdrive("1-3Y7TLZ5EpYYZmFLkRKjV508kcBeibjU", f"{root}/_downloads/rain_test/Rain100L.zip")
    ex = unzip(z, f"{root}/_extract/rain_test")
    src = find_dir(ex, "Rain100L")
    move_all(glob.glob(os.path.join(src, "rainy", "rain-*.png")), f"{dest}/input")
    # target must have the same file name as the input (DerainDehazeDataset._get_gt_path)
    move_all(glob.glob(os.path.join(src, "norain-*.png")), f"{dest}/target", lambda n: n.replace("norain-", "rain-"))
    shutil.rmtree(ex), os.remove(z)
    log(f"rain_test: {count(dest + '/input')} input, {count(dest + '/target')} target")


def step_urban100(root):
    dest = os.path.join(root, "test/denoise/urban100")
    if count(dest) == 100:
        return log("urban100: done")
    z = gdrive("1UlNulSoyflrEObwu19BBlT7f2_Wxycga", f"{root}/_downloads/urban100/urban100.zip")
    ex = unzip(z, f"{root}/_extract/urban100")
    move_all(glob.glob(os.path.join(find_dir(ex, "urban100"), "*.png")), dest)
    shutil.rmtree(ex), os.remove(z)
    log(f"urban100: {count(dest)} images")


def step_bsd68(root):
    dest = os.path.join(root, "test/denoise/bsd68")
    if count(dest) == 68:
        return log("bsd68: done")
    tmp = f"{root}/_downloads/cbsd68"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                    "https://github.com/clausmichele/CBSD68-dataset.git", tmp], check=True)
    subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "CBSD68/original"], check=True)
    move_all(glob.glob(os.path.join(tmp, "CBSD68/original/*.jpg")), dest)
    shutil.rmtree(tmp)
    log(f"bsd68: {count(dest)} images")


def step_sots(root):
    dest = os.path.join(root, "test/dehaze")
    if count(f"{dest}/input") == 500 and count(f"{dest}/target") == 492:
        return log("sots: done")
    z = curl(BOX.format(share=SOTS_SHARE, fid="766452360255"), f"{root}/_downloads/sots/SOTS.zip")
    ex = unzip(z, f"{root}/_extract/sots")
    out = find_dir(ex, "outdoor")
    move_all(glob.glob(os.path.join(out, "hazy", "*")), f"{dest}/input")
    move_all(glob.glob(os.path.join(out, "gt", "*.png")), f"{dest}/target")
    shutil.rmtree(ex), os.remove(z)
    log(f"sots: {count(dest + '/input')} hazy, {count(dest + '/target')} gt")


OTS_PARTS = {"part1": ("779003478999", 18200), "part2": ("779004528206", 18200),
             "part3": ("779004033141", 18200), "part4": ("779003792417", 17535)}


def step_ots(root):
    dest = os.path.join(root, "Train/Dehaze")
    if count(f"{dest}/original") != 2061:
        z = curl(BOX.format(share=OTS_SHARE, fid="778987136899"), f"{root}/_downloads/ots/clear.zip")
        ex = unzip(z, f"{root}/_extract/ots_clear")
        # keep .jpg: _get_nonhazy_name reuses the hazy image's extension
        move_all(glob.glob(os.path.join(find_dir(ex, "clear"), "*.jpg")), f"{dest}/original")
        shutil.rmtree(ex), os.remove(z)
        log(f"ots clear: {count(dest + '/original')}")
    for part, (fid, n) in OTS_PARTS.items():
        pdest = f"{dest}/synthetic/{part}"
        if count(pdest) == n:
            log(f"ots {part}: done")
            continue
        z = curl(BOX.format(share=OTS_SHARE, fid=fid), f"{root}/_downloads/ots/{part}.zip")
        ex = unzip(z, f"{root}/_extract/ots_{part}")
        move_all(glob.glob(os.path.join(find_dir(ex, part), "*.jpg")), pdest)
        shutil.rmtree(ex), os.remove(z)
        log(f"ots {part}: {count(pdest)} / {n}")


def to_png(src, dst):
    from PIL import Image
    Image.open(src).convert("RGB").save(dst)


def step_ohaze(root):
    dest = os.path.join(root, "test/dehaze_ohaze")
    if count(f"{dest}/input") == 45 and count(f"{dest}/target") == 45:
        return log("ohaze: done")
    z = curl("https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip", f"{root}/_downloads/ohaze/O-HAZE.zip")
    ex = unzip(z, f"{root}/_extract/ohaze")
    os.makedirs(f"{dest}/input", exist_ok=True), os.makedirs(f"{dest}/target", exist_ok=True)
    for f in glob.glob(os.path.join(find_dir(ex, "hazy"), "*")):
        shutil.move(f, f"{dest}/input/{os.path.basename(f).split('.')[0]}.jpg")  # NN_outdoor_hazy.jpg
    for f in glob.glob(os.path.join(find_dir(ex, "GT"), "*")):
        to_png(f, f"{dest}/target/{os.path.basename(f).split('_')[0]}.png")  # loader wants target/<NN>.png
    shutil.rmtree(ex), os.remove(z)
    log(f"ohaze: {count(dest + '/input')} / {count(dest + '/target')}")


def step_nhhaze(root):
    dest = os.path.join(root, "test/dehaze_nhhaze")
    if count(f"{dest}/input") == 55 and count(f"{dest}/target") == 55:
        return log("nhhaze: done")
    z = curl("https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip",
             f"{root}/_downloads/nhhaze/NH-HAZE.zip")
    ex = unzip(z, f"{root}/_extract/nhhaze")
    src = find_dir(ex, "NH-HAZE")
    move_all(glob.glob(os.path.join(src, "*_hazy.png")), f"{dest}/input")
    move_all(glob.glob(os.path.join(src, "*_GT.png")), f"{dest}/target", lambda n: n.replace("_GT", ""))
    shutil.rmtree(ex), os.remove(z)
    log(f"nhhaze: {count(dest + '/input')} / {count(dest + '/target')}")


CDD_WANT = {"train": 1183, "test": 200}


def step_cdd11(root):
    for split, n in CDD_WANT.items():
        if all(count(f"{root}/CDD-11/{split}/{d}") == n for d in ("clear", "haze_rain")):
            log(f"cdd11 {split}: done")
            continue
        rf = RangeFile(CDD.format(split=split))
        z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=4 << 20))
        members = [i for i in z.infolist() if not i.is_dir() and i.filename.split("/")[-2] in ("clear", "haze_rain")]
        log(f"cdd11 {split}: reading {len(members)} files from a {rf.size / 1e9:.1f} GB zip")
        for k, info in enumerate(members):
            sub = info.filename.split("/")[-2]
            out = f"{root}/CDD-11/{split}/{sub}/{os.path.basename(info.filename)}"
            if os.path.exists(out) and os.path.getsize(out) == info.file_size:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(info) as src, open(out + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(out + ".part", out)
            if k % 200 == 0:
                log(f"cdd11 {split}: {k}/{len(members)}")
        log(f"cdd11 {split}: clear {count(f'{root}/CDD-11/{split}/clear')}, "
            f"haze_rain {count(f'{root}/CDD-11/{split}/haze_rain')}")


STEPS = {"bsd400": step_bsd400, "urban100": step_urban100, "rain_train": step_rain_train,
         "rain_test": step_rain_test, "bsd68": step_bsd68, "sots": step_sots, "wed": step_wed,
         "ots": step_ots, "ohaze": step_ohaze, "nhhaze": step_nhhaze, "cdd11": step_cdd11}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data"))
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    args = ap.parse_args()
    failed = []
    for name in args.steps:
        try:
            STEPS[name](args.root)
        except Exception as e:
            log(f"{name} FAILED: {e}")
            failed.append(name)
    for d in ("_downloads", "_extract"):  # drop empty leftovers
        p = os.path.join(args.root, d)
        for sub, _, _ in sorted(os.walk(p), key=lambda t: -len(t[0])):
            if not any(os.scandir(sub)):
                os.rmdir(sub)
    log("all done" if not failed else f"failed steps: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
