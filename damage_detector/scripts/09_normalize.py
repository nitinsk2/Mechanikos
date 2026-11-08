"""Step 9: build the final dataset: every image as a 512 x 512 JPEG (quality 90) plus one CSV.
Run from the project root after 08_check_returns.py (re-run whenever more images arrive;
images already converted are skipped):
    python scripts/09_normalize.py

Inputs
  staging/real_damaged/ + manifests/real_damaged.csv + manifests/real_damaged_meta.csv  (CarDD)
  staging/real_base/    + manifests/base_meta.csv                                       (bases)
  returns/ai_<gen>/     + outputs*.csv, minus files listed in manifests/check_<gen>.csv
Output
  data/<group>/*.jpg and manifests/dataset.csv with one row per image:
  file, src, label (1 = AI, 0 = real, -1 = round trip), kind, generator, base_id, mask_file,
  feather_px, mask_frac, split, lighting, color, pre_jpeg

Rules
  * Every image is resized at most once (LANCZOS, short side 512, center crop) and saved as
    JPEG once at quality 90, with no EXIF. Bases and edits are already 512 x 512.
  * SDXL is the held-out generator: all its images are forced into the test split.
  * --attacker-jpeg (default 0.5): that share of AI images gets one extra JPEG save at a random
    quality 75-95 first, the way an attacker would save their edit before uploading it. Real
    photos are already JPEG-compressed at the source, so without this the model could learn
    "compressed once = AI". The quality used is recorded in pre_jpeg.
"""
import argparse
import csv
import hashlib
import io
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

M, DATA, RET = Path("manifests"), Path("data"), Path("returns")
GENS = ("sd15", "sdxl", "flux", "kontext")
HELD_OUT = "sdxl"
FIELDS = ["file", "src", "label", "kind", "generator", "base_id", "mask_file", "feather_px",
          "mask_frac", "split", "lighting", "color", "pre_jpeg"]


def read(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def center512(im):
    w, h = im.size
    if (w, h) == (512, 512):
        return im
    s = 512 / min(w, h)
    im = im.resize((max(512, round(w * s)), max(512, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - 512) // 2, (h - 512) // 2
    return im.crop((l, t, l + 512, t + 512))


def rnd(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def save(src, dst, pre_q=0):
    if dst.exists():
        return
    im = center512(ImageOps.exif_transpose(Image.open(src)).convert("RGB"))
    if pre_q:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=pre_q)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=90)       # no exif argument: metadata is dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker-jpeg", type=float, default=0.5)
    args = ap.parse_args()
    rows = []

    meta_rd = {r["file"]: r for r in read(M / "real_damaged_meta.csv")}
    for r in read(M / "real_damaged.csv"):
        src = Path("staging/real_damaged") / r["file"]
        m = meta_rd.get(r["file"])
        if not src.exists() or m is None:
            continue
        dst = DATA / "real_damaged" / (Path(r["file"]).stem + ".jpg")
        save(src, dst)
        rows.append(dict(file=dst.as_posix(), src=src.as_posix(), label=0, kind="real_damaged",
                         generator="real", split=m["split"], lighting=m["lighting"], color=m["color"],
                         pre_jpeg=0))

    base = {r["base_id"]: r for r in read(M / "base_meta.csv")}
    for bid, b in base.items():
        src = Path("staging/real_base") / f"{bid}.png"
        if not src.exists():
            continue
        dst = DATA / "real_base" / f"{bid}.jpg"
        save(src, dst)
        rows.append(dict(file=dst.as_posix(), src=src.as_posix(), label=0, kind="base_original",
                         generator="real", base_id=bid, split=b["split"], lighting=b["lighting"],
                         color=b["color"], pre_jpeg=0))

    plan = {r["mask_file"]: r for r in read(M / "edit_masks.csv")}
    for g in GENS:
        d = RET / f"ai_{g}"
        if not d.exists():
            continue
        skip = {r["file"] for r in read(M / f"check_{g}.csv")}
        outs = [r for c in sorted(d.glob("outputs*.csv")) for r in read(c)]
        if outs and not (M / f"check_{g}.csv").exists():
            print(f"!! run 08_check_returns.py --gen {g} first; skipping {g}")
            continue
        seen = set()
        for r in outs:
            name = Path(r["file"]).name
            src = d / name
            if name in skip or name in seen or not src.exists():
                continue
            seen.add(name)
            kind = r["kind"]
            label = -1 if kind == "round_trip" else 1
            if kind == "full_gen":
                split, light, color, mf, frac = r.get("split") or "train", r["lighting"], r["color"], "", ""
            else:
                b = base.get(r["base_id"])
                if b is None:
                    continue
                split, light, color = b["split"], b["lighting"], b["color"]
                mf = r.get("mask_file", "") if kind in ("paste_back", "whole_edit") else ""
                frac = plan.get(mf, {}).get("mask_frac", "")
                if kind == "paste_back" and mf.startswith("masks/"):      # mask detected from the edit
                    mf = (d / mf).as_posix()
                    frac = round(float((np.asarray(Image.open(mf).convert("L")) > 127).mean()), 4)
            if g == HELD_OUT:
                split = "test"
            pre_q = 0
            if label == 1 and rnd("pre" + name) < args.attacker_jpeg:
                pre_q = 75 + int(rnd("q" + name) * 21)
            dst = DATA / f"ai_{g}" / (Path(name).stem + ".jpg")
            save(src, dst, pre_q)
            rows.append(dict(file=dst.as_posix(), src=src.as_posix(), label=label, kind=kind,
                             generator=g, base_id=r.get("base_id", ""), mask_file=mf,
                             feather_px=r.get("feather_px", ""), mask_frac=frac, split=split,
                             lighting=light, color=color, pre_jpeg=pre_q))

    with open(M / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images -> manifests/dataset.csv")
    tally = Counter((r["kind"], r["generator"], r["split"]) for r in rows)
    for (k, g, s), n in sorted(tally.items()):
        print(f"   {k:14s} {g:5s} {s:5s} {n}")


if __name__ == "__main__":
    main()
