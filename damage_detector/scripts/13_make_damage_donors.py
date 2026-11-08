"""Backup plan for when no model paints visible damage: cut real damage patches out of CarDD
TRAINING photos, to seed the masks before the AI repaints them (gen_edits.py --seed-damage).
Run from the project root:
    python scripts/13_make_damage_donors.py
    python scripts/13_make_damage_donors.py --coco "raw/roboflow/cardd/annotations/instances_train2017.json"

It looks for CarDD's COCO annotation files (instances_*.json) under raw/ unless you pass --coco,
and crops each damage box (with a little margin) from the matching photo in staging/real_damaged/.
Only photos in the TRAIN split are used, so no validation or test photo leaks into the edits.
If no annotation file is found, it falls back to the center of train photos (CarDD photos are
close-ups of damage, so the center usually contains it). Then zip donors/ and upload it.
Outputs: donors/donor_0000.png ... (at most --n, longest side 256 px)
"""
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageOps

ap = argparse.ArgumentParser()
ap.add_argument("--coco", nargs="*", default=None)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--out", default="donors")
args = ap.parse_args()
M, REAL, OUT = Path("manifests"), Path("staging/real_damaged"), Path(args.out)
OUT.mkdir(exist_ok=True)


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


split = {r["file"]: r["split"] for r in read(M / "real_damaged_meta.csv")}
by_source = {}
for r in read(M / "real_damaged.csv"):
    if split.get(r["file"]) == "train" and (REAL / r["file"]).exists():
        by_source.setdefault(r.get("source_file", r["file"]), r["file"])

jsons = [Path(p) for p in args.coco] if args.coco else sorted(Path("raw").rglob("instances_*.json"))
crops = []
for jp in jsons:
    data = json.loads(Path(jp).read_text(encoding="utf-8"))
    names = {im["id"]: im["file_name"].split("/")[-1] for im in data["images"]}
    for a in data["annotations"]:
        f = by_source.get(names.get(a["image_id"], ""))
        x, y, w, h = a["bbox"]
        if f and min(w, h) >= 48:
            crops.append((f, x, y, w, h))
print(f"{len(jsons)} annotation files, {len(crops)} damage boxes in train photos")

random.seed(0)
random.shuffle(crops)
n = 0
if crops:
    for f, x, y, w, h in crops:
        im = ImageOps.exif_transpose(Image.open(REAL / f)).convert("RGB")
        mx, my = 0.1 * w, 0.1 * h
        box = (max(0, int(x - mx)), max(0, int(y - my)),
               min(im.width, int(x + w + mx)), min(im.height, int(y + h + my)))
        patch = im.crop(box)
        patch.thumbnail((256, 256))
        patch.save(OUT / f"donor_{n:04d}.png")
        n += 1
        if n >= args.n:
            break
else:
    print("no annotations found: using the center of train photos instead")
    files = list(by_source.values())
    random.shuffle(files)
    for f in files[:args.n]:
        im = ImageOps.exif_transpose(Image.open(REAL / f)).convert("RGB")
        w, h = im.size
        patch = im.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        patch.thumbnail((256, 256))
        patch.save(OUT / f"donor_{n:04d}.png")
        n += 1
print(f"{n} damage patches -> {OUT}/  (zip this folder and upload it to Drive)")
