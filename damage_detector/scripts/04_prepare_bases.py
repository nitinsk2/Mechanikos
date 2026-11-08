"""Step 4d: turn reviewed base photos into the 512x512 lossless PNGs that B and C edit.
Run from the project root (re-run any time you add photos; existing base_ids are kept):
    python scripts/04_prepare_bases.py

Inputs
  raw/phone/<car_folder>/*      one folder per physical car (folder name = car_id)
  raw/wikimedia/*               files still present after your review
  manifests/wikimedia.csv       licence + author per Wikimedia file (from 03_fetch_wikimedia.py)
  manifests/plate_blur.json     optional plate boxes (from blur_plates.py)
Outputs
  staging/real_base/base_0001.png ...   upright, plate-blurred, short side resized to 512 with
                                        LANCZOS, centre-cropped to 512x512, saved as PNG
  manifests/base_photos.csv             base_id, source, car_id, licence, attribution, blur flag

Because every base is already 512x512, the car mask, the edit masks, the paste-back and the
final JPEG save all live on the same pixel grid. The final normalize step for these images is
then just "save as JPEG q90", which keeps it to exactly one JPEG save.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image, ImageFilter, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("pillow-heif not installed: .heic phone photos will be skipped")

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
PHONE, WIKI = Path("raw/phone"), Path("raw/wikimedia")
WIKI_MANIFEST = Path("manifests/wikimedia.csv")
BLUR = Path("manifests/plate_blur.json")
OUT = Path("staging/real_base")
MANIFEST = Path("manifests/base_photos.csv")
FIELDS = ["base_id", "file", "source", "source_file", "car_id", "license", "license_url",
          "author", "page_url", "plate_blurred", "orig_w", "orig_h", "phash"]
SIZE, PHASH_MAX_DIST = 512, 6


def center_square(im, size=SIZE):
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


def blur_boxes(im, boxes):
    W, H = im.size
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        region = im.crop((x0, y0, x1, y1))
        r = max(4, min(x1 - x0, y1 - y0) // 3)
        im.paste(region.filter(ImageFilter.GaussianBlur(r)), (x0, y0))
    return im


def collect_sources():
    wiki_meta = {}
    if WIKI_MANIFEST.exists():
        with open(WIKI_MANIFEST, encoding="utf-8") as f:
            wiki_meta = {r["file"]: r for r in csv.DictReader(f)}
    items = []
    for p in sorted(PHONE.rglob("*")):
        if p.suffix.lower() in EXTS:
            car = p.parent.name if p.parent != PHONE else p.stem
            items.append(dict(path=p, source="phone", car_id=f"phone_{car}",
                              license="team-owned photo", license_url="", author="team",
                              page_url=""))
    for p in sorted(WIKI.glob("*")):
        if p.suffix.lower() in EXTS:
            m = wiki_meta.get(p.name)
            if m is None:
                print(f"   skip {p.name}: not in {WIKI_MANIFEST} (licence unknown)")
                continue
            items.append(dict(path=p, source="wikimedia", car_id=f"wm_{m['pageid']}",
                              license=m["license"], license_url=m["license_url"],
                              author=m["author"], page_url=m["page_url"]))
    return items


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blur = json.loads(BLUR.read_text()) if BLUR.exists() else {}
    old = {}
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            old = {r["source_file"]: r for r in csv.DictReader(f)}
    next_id = 1 + max((int(r["base_id"].split("_")[1]) for r in old.values()), default=0)

    rows, hashes, skipped = [], [], Counter()
    for it in collect_sources():
        key = it["path"].as_posix()
        try:
            im = ImageOps.exif_transpose(Image.open(it["path"])).convert("RGB")
        except Exception:  # noqa: BLE001
            skipped["unreadable"] += 1
            continue
        if min(im.size) < SIZE:
            skipped["short side < 512"] += 1
            continue
        boxes = blur.get(key, [])
        ow, oh = im.size
        sq = center_square(blur_boxes(im, boxes))
        ph = imagehash.phash(sq)
        if any(ph - h <= PHASH_MAX_DIST for h in hashes):
            skipped["near duplicate"] += 1
            continue
        hashes.append(ph)
        if key in old:
            base_id = old[key]["base_id"]
        else:
            base_id, next_id = f"base_{next_id:04d}", next_id + 1
        sq.save(OUT / f"{base_id}.png", optimize=False, compress_level=1)
        rows.append(dict(base_id=base_id, file=f"{base_id}.png", source=it["source"],
                         source_file=key, car_id=it["car_id"], license=it["license"],
                         license_url=it["license_url"], author=it["author"],
                         page_url=it["page_url"], plate_blurred=int(bool(boxes)),
                         orig_w=ow, orig_h=oh, phash=str(ph)))

    # remove PNGs whose source was deleted since the last run
    live = {r["file"] for r in rows}
    for p in OUT.glob("base_*.png"):
        if p.name not in live:
            p.unlink()

    rows.sort(key=lambda r: r["base_id"])
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    print(f"{len(rows)} base photos -> {OUT}/  (manifest: {MANIFEST})")
    print("by source:", dict(Counter(r["source"] for r in rows)))
    print("distinct cars:", len({r["car_id"] for r in rows}),
          "| plate-blurred:", sum(r["plate_blurred"] for r in rows))
    if skipped:
        print("skipped:", dict(skipped))


if __name__ == "__main__":
    main()
