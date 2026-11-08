"""Step 7 (before zipping the hand-off): assign splits, slice labels and final prompts.
Run from the project root (damage_detector):
    python scripts/07_finalize_handoff.py

What it does
  * Splits every base photo 70/15/15 into train / val / test by a hash of its car_id, so a
    car and all its edits always land in the same split, on every machine, every run.
  * Splits the real damaged photos the same way by source_group.
  * Labels each base photo's lighting (dark / normal / bright, from the car region's brightness)
    and car color (white / black / silver/gray / red / blue / other, from the car body).
  * Adds split, lighting and color columns to manifests/edit_masks.csv and puts the car color
    into each prompt ("silver car, dented car door ..."), so all three generators get the same,
    more specific prompt. The original prompt is kept in prompt_base. Safe to re-run.

Outputs
  manifests/base_meta.csv            base_id, car_id, split, lighting, color
  manifests/real_damaged_meta.csv    file, source_group, split, lighting, color
  manifests/edit_masks.csv           updated in place (split, lighting, color, prompt_base)
"""
import csv
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

BASES = Path("staging/real_base")
CAR = Path("masks/car")
REAL = Path("staging/real_damaged")
M = Path("manifests")
COLOR_WORD = {"white": "white", "black": "black", "silver/gray": "silver",
              "red": "red", "blue": "blue", "other": ""}


def split_of(key: str) -> str:
    k = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if k < 70 else ("val" if k < 85 else "test")


def lighting_of(v_values) -> str:
    v = float(np.mean(v_values))
    return "dark" if v < 90 else ("bright" if v > 170 else "normal")


def color_of(hsv_pixels) -> str:
    h, s, v = np.median(hsv_pixels, axis=0)          # OpenCV hue is 0-179
    if v < 55:                                        # very dark: saturation is unreliable
        return "black"
    if s < 40:
        return "white" if v > 180 else ("black" if v < 60 else "silver/gray")
    if h < 10 or h >= 160:
        return "red"
    if 95 <= h < 130:
        return "blue"
    return "other"


def body_band(car):
    """Middle-lower band of the car (doors, panels): skips windows, roof and most of the tires."""
    ys, _ = np.nonzero(car)
    if len(ys) == 0:
        return car
    top, bot = ys.min(), ys.max()
    band = np.zeros_like(car)
    band[int(top + 0.40 * (bot - top)):int(top + 0.85 * (bot - top)) + 1] = True
    inner = cv2.erode(car.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    out = inner & band
    return out if out.sum() > 200 else car


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    car_id = {r["base_id"]: (r.get("car_id") or r["base_id"]) for r in read(M / "base_photos.csv")}
    edits = read(M / "edit_masks.csv")

    # base photos: split by car_id, lighting and color from the car mask
    meta = {}
    for bid in sorted({r["base_id"] for r in edits}):
        im = np.asarray(Image.open(BASES / f"{bid}.png").convert("RGB"))
        car = np.asarray(Image.open(CAR / f"{bid}.png").convert("L")) > 127
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        region = car if car.sum() > 200 else np.ones(car.shape, bool)
        cid = car_id.get(bid, bid)
        meta[bid] = dict(base_id=bid, car_id=cid, split=split_of(cid),
                         lighting=lighting_of(hsv[..., 2][region]),
                         color=color_of(hsv[body_band(region)]))
    write(M / "base_meta.csv", list(meta.values()), ["base_id", "car_id", "split", "lighting", "color"])

    # edit masks: add split / slice labels, put the car color in the prompt (idempotent)
    for r in edits:
        m = meta[r["base_id"]]
        r["split"], r["lighting"], r["color"] = m["split"], m["lighting"], m["color"]
        base_prompt = r.get("prompt_base") or r["prompt"]
        r["prompt_base"] = base_prompt
        word = COLOR_WORD[m["color"]]
        r["prompt"] = f"{word} car, {base_prompt}" if word else base_prompt
    fields = list(dict.fromkeys(list(edits[0].keys())))
    write(M / "edit_masks.csv", edits, fields)

    # real damaged photos: split by source_group, approximate lighting and color
    real_rows = []
    rd_path = M / "real_damaged.csv"
    if rd_path.exists():
        for r in read(rd_path):
            p = REAL / r["file"]
            if not p.exists():
                continue                      # deleted during review
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.thumbnail((256, 256))
            hsv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HSV)
            h, w = hsv.shape[:2]
            center = hsv[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3)
            grp = r.get("source_group") or r["file"]
            real_rows.append(dict(file=r["file"], source_group=grp, split=split_of("rd_" + grp),
                                  lighting=lighting_of(hsv[..., 2]), color=color_of(center)))
        write(M / "real_damaged_meta.csv", real_rows, ["file", "source_group", "split", "lighting", "color"])

    print("base photos by split:", dict(Counter(m["split"] for m in meta.values())))
    print("edits by split:      ", dict(Counter(r["split"] for r in edits)))
    print("base lighting:       ", dict(Counter(m["lighting"] for m in meta.values())))
    print("base color:          ", dict(Counter(m["color"] for m in meta.values())))
    if real_rows:
        print("real damaged by split:", dict(Counter(r["split"] for r in real_rows)))
    print("example prompt:", edits[0]["prompt"])
    print("Now rebuild handoff_A.zip (see the runbook) and upload it.")


if __name__ == "__main__":
    main()
