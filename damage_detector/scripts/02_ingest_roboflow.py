"""Step 3: clean the Roboflow exports into the real-damaged pool.
Run from the project root after unzipping each export into raw/roboflow/<slug>/:
    python scripts/02_ingest_roboflow.py

What it does, per image:
  * collapses Roboflow augmentation copies (same source name before ".rf.") to one image
  * drops images whose short side is < 512 px (upscaling only real photos is a shortcut)
  * drops grayscale images and images with flat letterbox bars on the edges
  * removes near-duplicates across ALL sets with a perceptual hash (Hamming distance <= 6)
  * copies survivors (bytes untouched) to staging/real_damaged/ and writes
    manifests/real_damaged.csv with source, license and attribution for every file
Nothing is resized or re-saved here: the shared normalize script does that later, once.
"""
import csv
import re
import shutil
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

# Edit this table if you add or swap a dataset. slug = folder name under raw/roboflow/
DATASETS = {
    "cardd": dict(
        version="CarDD_release", license="CarDD license (permission granted to team)",
        author="PIC Lab, USTC - Wang et al., CarDD (cite the paper)",
        url="https://cardd-ustc.github.io/"),
}

RAW = Path("raw/roboflow")
OUT = Path("staging/real_damaged")
MANIFEST = Path("manifests/real_damaged.csv")
MIN_SHORT_SIDE = 512
PHASH_MAX_DIST = 6
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def source_stem(name: str) -> str:
    """'IMG_0042_jpg.rf.9f8e7d.jpg' -> 'img_0042'"""
    stem = name.split(".rf.")[0]
    return re.sub(r"_(jpe?g|png|webp|bmp)$", "", stem, flags=re.I).lower()


def is_grayscale(a: np.ndarray) -> bool:
    small = a[::8, ::8].astype(np.int16)
    return int(np.abs(small[..., 0] - small[..., 1]).max()) < 4 and \
        int(np.abs(small[..., 1] - small[..., 2]).max()) < 4


def has_letterbox(a: np.ndarray, band: int = 6) -> bool:
    g = a.mean(axis=2)
    for strip in (g[:band], g[-band:], g[:, :band], g[:, -band:]):
        if strip.std() < 2.0 and (strip.mean() < 12 or strip.mean() > 243):
            return True
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    kept, seen_groups, hashes = [], set(), []
    drops = {"augment_copy": 0, "too_small": 0, "grayscale": 0, "letterbox": 0,
             "near_duplicate": 0, "unreadable": 0}

    for slug, meta in DATASETS.items():
        files = sorted(p for p in (RAW / slug).rglob("*") if p.suffix.lower() in EXTS)
        if not files:
            print(f"!! no images under {RAW / slug} - check the folder name")
            continue
        n_before = len(kept)
        for p in files:
            group = (slug, source_stem(p.name))
            if group in seen_groups:
                drops["augment_copy"] += 1
                continue
            seen_groups.add(group)
            try:
                im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            except Exception:  # noqa: BLE001
                drops["unreadable"] += 1
                continue
            w, h = im.size
            if min(w, h) < MIN_SHORT_SIDE:
                drops["too_small"] += 1
                continue
            a = np.asarray(im)
            if is_grayscale(a):
                drops["grayscale"] += 1
                continue
            if has_letterbox(a):
                drops["letterbox"] += 1
                continue
            ph = imagehash.phash(im)
            if any(ph - other <= PHASH_MAX_DIST for other in hashes):
                drops["near_duplicate"] += 1
                continue
            hashes.append(ph)
            new_name = f"rd_{len(kept):05d}{p.suffix.lower()}"
            shutil.copy2(p, OUT / new_name)
            split_in_src = next((s for s in ("train", "valid", "test") if s in p.parts), "")
            kept.append(dict(file=new_name, dataset=slug, version=meta["version"],
                             source_split=split_in_src, source_file=p.name,
                             source_group=group[1], width=w, height=h, phash=str(ph),
                             license=meta["license"], author=meta["author"], url=meta["url"]))
        print(f"{slug}: {len(files)} files -> kept {len(kept) - n_before}")

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        if kept:
            wr = csv.DictWriter(f, fieldnames=list(kept[0]))
            wr.writeheader()
            wr.writerows(kept)
    print(f"\nKEPT {len(kept)} real damaged photos -> {OUT}/ (manifest: {MANIFEST})")
    print("dropped:", ", ".join(f"{k}={v}" for k, v in drops.items()))
    if len(kept) < 800:
        print("!! under ~800: add a third CC BY 4.0 / MIT set (see runbook section 3.5)")


if __name__ == "__main__":
    main()
