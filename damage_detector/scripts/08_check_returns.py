"""Step 8: check what B and C send back, before any of it enters the dataset.
Download each generator's Drive folder into returns/ (returns/ai_sd15, returns/ai_sdxl,
returns/ai_flux), then run from the project root:

    python scripts/08_check_returns.py --gen sd15
    python scripts/08_check_returns.py --gen sdxl
    python scripts/08_check_returns.py --gen flux

Works on partial batches (e.g. the first 10 rows). Writes:
  manifests/check_<gen>.csv           one line per problem (09_normalize.py skips these files)
  manifests/done_<gen>.txt            edit masks with a paste-back that passed every check
  manifests/missing_<gen>.txt         edit masks still missing or failing, for this generator's splits
  manifests/check_preview_<gen>.jpg   base | mask | edit | change map, for random paste-backs

Checks: outputs CSVs exist with the agreed columns and match the files; every image is a true PNG,
512 x 512; names follow the rules; paste-backs use the mask, prompt and feather from
edit_masks.csv, change nothing outside the feathered mask and change enough inside it;
whole edits are NOT pasted back; round trips are close to the base but not identical;
fully generated images aren't blank, have color/lighting, and aren't near-duplicates.
"""
import argparse
import csv
import math
import random
import re
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageDraw

BASES = Path("staging/real_base")
EDIT = Path("masks/edit")
M = Path("manifests")
RETURNS = Path("returns")
COLS = ["file", "kind", "base_id", "mask_file", "prompt", "feather_px", "model", "seed",
        "color", "lighting", "generator", "split"]
KINDS = {"paste_back", "whole_edit", "round_trip", "full_gen"}
SPLITS = {"sd15": {"train", "val", "test"}, "flux": {"train", "val", "test"}, "sdxl": {"test"},
          "kontext": {"train", "val", "test"}}
OUTSIDE_TOL = 2     # max pixel change allowed outside the feathered mask (rounding only)
WEAK_EDIT = 8       # mean change inside the mask below this: the generator barely did anything
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=["sd15", "sdxl", "flux", "kontext"])
    ap.add_argument("--dir", default=None, help="defaults to returns/ai_<gen>")
    ap.add_argument("--preview", type=int, default=12)
    args = ap.parse_args()
    g = args.gen
    d = Path(args.dir) if args.dir else RETURNS / f"ai_{g}"
    issues = []

    def bad(name, why):
        issues.append({"file": name, "problem": why})
        return False

    with open(M / "edit_masks.csv", encoding="utf-8") as f:
        plan = {r["mask_file"]: r for r in csv.DictReader(f)}
    expected = {mf for mf, r in plan.items() if r.get("split", "train") in SPLITS[g]}

    csvs = sorted(d.glob("outputs*.csv"))
    if not csvs:
        raise SystemExit(f"no outputs*.csv in {d}: download the whole Drive folder, CSVs included")
    rows = []
    for c in csvs:
        with open(c, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            miss = [x for x in COLS if x not in (rd.fieldnames or [])]
            if miss:
                bad(c.name, f"missing columns: {miss}")
            rows += list(rd)

    kinds, listed, ok_masks, hashes, previews = Counter(), set(), set(), [], []
    for r in rows:
        name = Path(r.get("file", "").strip()).name
        kind = r.get("kind", "").strip()
        if name in listed:          # two people met on the same row: keep one listing
            continue
        listed.add(name)
        kinds[kind] += 1
        p = d / name
        if kind not in KINDS:
            bad(name, f"unknown kind '{kind}'")
            continue
        if not p.exists():
            bad(name, "listed in an outputs CSV but the file isn't in the folder")
            continue
        with open(p, "rb") as fh:
            if fh.read(8) != PNG_MAGIC:
                bad(name, "not a real PNG (a JPEG save here would leak into the labels)")
                continue
        im = Image.open(p)
        if im.size != (512, 512):
            bad(name, f"size {im.size}, expected 512 x 512")
            continue
        if r.get("generator", "").strip() not in ("", g):
            bad(name, f"generator column says {r.get('generator')}, folder is {g}")
        out = rgb(p)

        if kind == "full_gen":
            if not re.fullmatch(rf"gen_{g}_\d{{4}}\.png", name):
                bad(name, f"name should look like gen_{g}_0000.png")
            if r.get("base_id", "").strip() or r.get("mask_file", "").strip():
                bad(name, "full_gen rows should have empty base_id and mask_file")
            if not r.get("color", "").strip() or not r.get("lighting", "").strip():
                bad(name, "color or lighting is empty (needed for the slice report)")
            if out.std() < 5:
                bad(name, "image is nearly blank")
            hashes.append((name, imagehash.phash(im.convert("RGB"))))
            continue

        bid = r.get("base_id", "").strip()
        bp = BASES / f"{bid}.png"
        if not bp.exists():
            bad(name, f"base_id '{bid}' is not one of our base photos (mock data?)")
            continue
        diff = np.abs(out - rgb(bp)).max(axis=2)

        if kind == "paste_back":
            mf = r.get("mask_file", "").strip()
            own = mf.startswith("masks/")          # kontext: mask detected from the edit, saved next to it
            key = f"{name[:-len(g) - 5]}.png" if own else mf     # the planned mask this row belongs to
            pr = plan.get(key)
            if pr is None:
                bad(name, f"mask_file '{mf}' is not in edit_masks.csv")
                continue
            good = True
            if pr["base_id"] != bid:
                good = bad(name, f"base_id {bid} doesn't match the mask's base {pr['base_id']}")
            if name != f"{key[:-4]}_{g}.png":
                good = bad(name, f"name should be {key[:-4]}_{g}.png")
            if not own and r.get("prompt", "").strip() != pr["prompt"].strip():
                good = bad(name, "prompt differs from edit_masks.csv (breaks the matched pairs)")
            if str(r.get("feather_px", "")).strip() != str(pr["feather_px"]).strip():
                good = bad(name, "feather_px differs from edit_masks.csv")
            mpath = d / mf if own else EDIT / mf
            if not mpath.exists():
                bad(name, f"mask {mf} is missing")
                continue
            m = np.asarray(Image.open(mpath).convert("L")) > 127
            rad = int(math.ceil(1.5 * float(pr["feather_px"] or 0))) + 2
            grown = cv2.dilate(m.astype(np.uint8), np.ones((2 * rad + 1, 2 * rad + 1), np.uint8)) > 0
            outside = int(diff[~grown].max()) if (~grown).any() else 0
            inside = float(diff[m].mean()) if m.any() else 0.0
            if outside > OUTSIDE_TOL:
                good = bad(name, f"pixels changed outside the mask (max {outside}): "
                                 "no paste-back, wrong base, or resized base")
            if inside < WEAK_EDIT:
                good = bad(name, f"edit is too weak inside the mask (mean change {inside:.1f})")
            if good:
                ok_masks.add(key)
            previews.append((name, p, bp, m, diff))

        elif kind == "whole_edit":
            mf = r.get("mask_file", "").strip()
            if name != f"{mf[:-4]}_whole_{g}.png":
                bad(name, f"name should be {mf[:-4]}_whole_{g}.png")
            if diff.mean() < 1:
                bad(name, "almost identical to the base: was it pasted back by mistake?")

        elif kind == "round_trip":
            if name != f"{bid}_rt_{g}.png":
                bad(name, f"name should be {bid}_rt_{g}.png")
            mean = float(diff.mean())
            if mean == 0:
                bad(name, "identical to the base: the round trip wasn't applied")
            elif mean > 20:
                bad(name, f"too different for a round trip (mean change {mean:.1f})")

    for p in sorted(d.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and p.name not in listed:
            bad(p.name, "file is in the folder but not in any outputs CSV")

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i][1] - hashes[j][1] <= 4:
                bad(hashes[j][0], f"near-duplicate of {hashes[i][0]}")

    missing = sorted(expected - ok_masks)
    (M / f"missing_{g}.txt").write_text("\n".join(missing), encoding="utf-8")
    (M / f"done_{g}.txt").write_text("\n".join(sorted(ok_masks)), encoding="utf-8")
    with open(M / f"check_{g}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "problem"])
        w.writeheader()
        w.writerows(issues)

    random.seed(0)
    pick = random.sample(previews, min(args.preview, len(previews)))
    if pick:
        t = 192
        sheet = Image.new("RGB", (4 * t, len(pick) * (t + 16)), "white")
        dr = ImageDraw.Draw(sheet)
        for k, (name, p, bp, m, diff) in enumerate(pick):
            base = Image.open(bp).convert("RGB")
            ov = np.asarray(base).copy()
            ov[m] = (0.5 * ov[m] + [127, 0, 0]).astype(np.uint8)
            change = Image.fromarray(np.clip(diff * 4, 0, 255).astype(np.uint8)).convert("RGB")
            tiles = [base, Image.fromarray(ov), Image.open(p).convert("RGB"), change]
            y = k * (t + 16)
            for c, tile in enumerate(tiles):
                sheet.paste(tile.resize((t, t)), (c * t, y))
            dr.text((4, y + t + 2), name, fill="black")
        sheet.save(M / f"check_preview_{g}.jpg", quality=90)

    by_split = Counter(plan[mf]["split"] for mf in ok_masks if "split" in plan[mf])
    print(f"{g}: {len(rows)} rows, by kind {dict(kinds)}")
    print(f"valid paste-backs: {len(ok_masks)} of {len(expected)} expected  by split {dict(by_split)}")
    if issues:
        print(f"{len(issues)} problems -> manifests/check_{g}.csv")
        tally = Counter(re.sub(r"\(.*?\)|'.*?'|\S+\.png|\d+", "#", i["problem"]) for i in issues)
        for prob, n in tally.most_common(10):
            print(f"   {n:4d}  {prob}")
    else:
        print("no problems found")
    if pick:
        print(f"look at manifests/check_preview_{g}.jpg: damage inside the red area, "
              "change map black outside it")
    done = {x: set(filter(None, (M / f"done_{x}.txt").read_text().split("\n")))
            for x in ("sd15", "sdxl", "flux", "kontext") if (M / f"done_{x}.txt").exists()}
    if len(done) > 1:
        print("matched pairs:", {f"{a}+{b}": len(done[a] & done[b])
                                 for a in done for b in done if a < b})


if __name__ == "__main__":
    main()
