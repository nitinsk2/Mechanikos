"""Step 5: place random damage-edit masks inside each car mask (the hand-off B and C inpaint).
Run from the project root after 05_check_bases.py:
    python scripts/06_make_edit_masks.py                    # 1 edit per base
    python scripts/06_make_edit_masks.py --edits-per-base 2 # more edits if bases are scarce

Inputs   staging/real_base/base_XXXX.png, masks/car/base_XXXX.png, manifests/base_car_check.csv
Outputs  masks/edit/base_XXXX_e1.png   512x512, 255 = region the generator repaints
         manifests/edit_masks.csv      one row per edit: mask file, size bucket, actual size,
                                       damage prompt, feather width for paste-back
         masks/edit_preview/*.jpg      red overlays of the first --preview edits, to eyeball

Each mask is one irregular blob, fully inside the car, covering 3-15% of the image. Sizes are
spread evenly over the plan's buckets (3-6%, 6-10%, 10-15%) so recall-by-edit-size has data;
bases whose car is 'small' only get 3-6%. Blob centres sit in the middle band of the car
(doors, bumpers, hood, fenders) rather than the roof line. The SAME mask, prompt and feather
are used by both generators, so the cross-generator test has matched pairs.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASES, CAR, OUT = Path("staging/real_base"), Path("masks/car"), Path("masks/edit")
PREVIEW = Path("masks/edit_preview")
CHECK, MANIFEST = Path("manifests/base_car_check.csv"), Path("manifests/edit_masks.csv")
BUCKETS = {"3-6": (0.03, 0.06), "6-10": (0.06, 0.10), "10-15": (0.10, 0.15)}
FEATHERS = [0, 2, 4, 6, 8]  # paste-back edge blur in px, varied so the seam isn't a shortcut
PROMPTS = [
    "deep dent and scraped paint on the car door, realistic phone photo",
    "dented car door with long scratches, realistic phone photo",
    "cracked and dented rear bumper, realistic phone photo",
    "cracked front bumper with scuffed paint, realistic phone photo",
    "crumpled fender with scraped paint, realistic phone photo",
    "dented hood with chipped paint, realistic phone photo",
    "long deep key scratch across the car panel, realistic phone photo",
    "large dent with paint transfer on the car side, realistic phone photo",
    "shattered headlight and dented surrounding panel, realistic phone photo",
    "rust-free fresh collision damage, bent metal panel, realistic phone photo",
]


def shape_fn(h, w, cy, cx, rng):
    """Return f(R) -> bool mask of an irregular blob of scale R centred at (cy, cx)."""
    yy, xx = np.mgrid[0:h, 0:w]
    aspect = rng.uniform(1.0, 2.2)          # wider than tall, like panel damage
    ang = rng.uniform(-0.4, 0.4)
    dy, dx = yy - cy, xx - cx
    ry = dy * np.cos(ang) - dx * np.sin(ang)
    rx = (dy * np.sin(ang) + dx * np.cos(ang)) / aspect
    th, r = np.arctan2(ry, rx), np.hypot(ry, rx)
    wobble = np.ones_like(th)
    for n in range(2, 7):
        wobble += rng.uniform(0, 0.22) * np.cos(n * th + rng.uniform(0, 2 * np.pi))
    return lambda R: r <= R * wobble


def largest_component(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def make_mask(car, target, rng, tries=12):
    h, w = car.shape
    inner = cv2.erode(car.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys, xs = np.nonzero(inner if inner.any() else car)
    if len(ys) == 0:
        return None
    top, bot = ys.min(), ys.max()
    band = (ys >= top + 0.35 * (bot - top)) & (ys <= top + 0.90 * (bot - top))
    if band.any():
        ys, xs = ys[band], xs[band]
    for _ in range(tries):
        i = rng.integers(len(ys))
        f = shape_fn(h, w, ys[i], xs[i], rng)
        lo, hi = 2.0, float(max(h, w))
        for _ in range(22):                 # binary search the blob scale
            mid = (lo + hi) / 2
            frac = (f(mid) & car).mean()
            lo, hi = (mid, hi) if frac < target else (lo, mid)
        m = largest_component(f(hi) & car)
        if abs(m.mean() - target) <= 0.25 * target:
            return m
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits-per-base", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)

    with open(CHECK, encoding="utf-8") as f:
        bases = [r for r in csv.DictReader(f) if r["status"] in ("ok", "small")]
    rng = np.random.default_rng(args.seed)
    rows, failed, k_bucket = [], 0, 0
    for b in bases:
        car = np.array(Image.open(CAR / f"{b['base_id']}.png")) > 127
        for k in range(1, args.edits_per_base + 1):
            if b["status"] == "small":
                bucket = "3-6"
            else:
                bucket = list(BUCKETS)[k_bucket % 3]
                k_bucket += 1
            target = rng.uniform(*BUCKETS[bucket])
            m = make_mask(car, target, rng)
            if m is None:
                failed += 1
                continue
            name = f"{b['base_id']}_e{k}.png"
            Image.fromarray((m * 255).astype(np.uint8)).save(OUT / name)
            rows.append(dict(mask_file=name, base_id=b["base_id"], edit_k=k, size_bucket=bucket,
                             target_frac=round(target, 4), mask_frac=round(float(m.mean()), 4),
                             prompt=PROMPTS[rng.integers(len(PROMPTS))],
                             feather_px=int(rng.choice(FEATHERS))))
            if len(rows) <= args.preview:
                im = np.array(Image.open(BASES / f"{b['base_id']}.png").convert("RGB"))
                im[m] = (0.5 * im[m] + [127, 0, 0]).astype(np.uint8)
                Image.fromarray(im).save(PREVIEW / name.replace(".png", ".jpg"), quality=85)

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["mask_file", "base_id", "edit_k", "size_bucket",
                                           "target_frac", "mask_frac", "prompt", "feather_px"])
        wr.writeheader()
        wr.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} edit masks for {len(bases)} bases -> {OUT}/  (manifest: {MANIFEST})")
    print("by size bucket:", dict(Counter(r["size_bucket"] for r in rows)))
    if failed:
        print(f"{failed} edits could not be placed (car too small or oddly shaped) - skipped")
    print(f"Check the overlays in {PREVIEW}/ : red should sit on body panels, not glass or road.")


if __name__ == "__main__":
    main()
