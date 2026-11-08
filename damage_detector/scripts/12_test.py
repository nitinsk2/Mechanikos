"""Step 12: the one and only test run. Do this once, at the agreed time, and report the numbers as
they come out. Run from the project root after 11_train.py:
    python scripts/12_test.py
    python scripts/12_test.py --dry-run     # pipeline test with fake features (no GPU)

Scores the test split (including its round trips) with each trained experiment, then re-scores the
test photos for the main model after JPEG quality 70, JPEG quality 50 and half-resolution.
Writes predictions/test_all.csv for B's eval_report.py, and prints the headline numbers.
Refuses to run twice unless you pass --force (re-running and picking the best run is cheating
yourself; if you must re-run after a real bug, say so in the write-up).
"""
import argparse
import importlib.util
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

F, MODELS, OUT = Path("features"), Path("models"), Path("predictions")
VARIANTS = ("jpeg70", "jpeg50", "half")

spec = importlib.util.spec_from_file_location("fx", Path(__file__).with_name("10_extract_features.py"))
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)


def variant(im, v):
    if v == "half":
        return im.resize((256, 256), Image.BILINEAR).resize((512, 512), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(v[4:]))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def norm(X):
    X = X.astype(np.float32)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def predict(m, X):
    s = m["clf"].predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(len(X), 9)
    p = m["cal"].predict_proba(s.max(1).reshape(-1, 1))[:, 1]
    return p, s.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    marker = OUT / ".test_done"
    if marker.exists() and not args.force:
        raise SystemExit("The test set was already used. Pass --force only to fix a real bug, "
                         "and mention the re-run in the write-up.")

    X = norm(np.load(F / "feats.npy"))
    idx = pd.read_csv(F / "index.csv", dtype=str, keep_default_na=False)
    idx["label"] = idx["label"].astype(int)
    sel = (idx.split == "test").values        # round trips included: they inherit their base's split
    test = idx[sel].reset_index(drop=True)
    Xt = X[sel]

    parts = []
    for name in ("main", "sd15_only", "flux_only"):
        path = MODELS / f"{name}.joblib"
        if not path.exists():
            continue
        m = joblib.load(path)
        p, top = predict(m, Xt)
        parts.append(test.assign(prob=p, flagged=p > m["threshold"], top_crop=top,
                                 experiment=name, variant="clean"))

    main_m = joblib.load(MODELS / "main.joblib")
    embed = fx.fake_embed if args.dry_run else fx.load_clip()
    rob = test[test.label >= 0].reset_index(drop=True)
    for v in VARIANTS:
        feats = []
        for i in tqdm(range(0, len(rob), 4), desc=v):
            ims = [variant(Image.open(f).convert("RGB"), v) for f in rob.file[i:i + 4]]
            feats.append(embed([fx.crops_of(im) for im in ims]))
        p, top = predict(main_m, norm(np.concatenate(feats)))
        parts.append(rob.assign(prob=p, flagged=p > main_m["threshold"], top_crop=top,
                                experiment="main", variant=v))

    pred = pd.concat(parts, ignore_index=True)
    pred.to_csv(OUT / "test_all.csv", index=False)
    marker.write_text("done")

    c = pred[(pred.experiment == "main") & (pred.variant == "clean")]
    print("\nMAIN MODEL (trained on FLUX.1 Kontext edits), test split, threshold from validation")
    print("share flagged, by kind and generator:")
    print(c.groupby(["kind", "generator"]).flagged.agg(["mean", "size"]).round(3).to_string())
    for name in ("sd15_only", "flux_only"):
        e = pred[(pred.experiment == name) & (pred.label == 1)]
        if len(e):
            print(f"\n{name}: paste-back recall by generator")
            print(e[e.kind == "paste_back"].groupby("generator").flagged.agg(["mean", "size"]).round(3).to_string())
    print("\nwrote predictions/test_all.csv -> send it to B for eval_report.py")


if __name__ == "__main__":
    main()
