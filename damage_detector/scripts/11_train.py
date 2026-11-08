"""Step 11: train the crop classifier, calibrate it and set the review threshold (validation only).
Run from the project root after 10_extract_features.py:
    python scripts/11_train.py

Experiments (AI images from these generators are used for training; real photos always are):
  main       FLUX.1 Kontext  -> the model the demo uses; SDXL (and FLUX Fill, if made) are held out
  sd15_only  SD 1.5          -> tested later on FLUX: old generator to a modern one
  flux_only  FLUX            -> tested later on SD 1.5
For each: logistic regression on L2-normalized crop features (class_weight="balanced", because
real crops outnumber AI crops), C picked by image-level ROC-AUC on validation, image score =
highest crop probability, Platt calibration on validation, and a threshold that flags at most
5% of real validation photos. The test split is never touched here.
Outputs: models/<experiment>.joblib, printed validation numbers.

--domain-balance (use this): the first test run showed the model partly learned the photo SOURCE
(CarDD close-ups score near 0, untouched Wikimedia car photos much higher), because every AI edit
was made on a Wikimedia photo. With this flag, real crops from untouched base photos get as much
total weight as all CarDD crops together, AI crops get as much as all real crops, and the review
threshold is set to flag at most 5% of the untouched base photos in validation (the hard,
same-source real photos), not of all real photos. Previous models are kept as *_v1.joblib.
"""
import argparse
import shutil
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

F, MODELS = Path("features"), Path("models")
EXPERIMENTS = {"main": ["kontext"], "sd15_only": ["sd15"], "flux_only": ["flux"]}
C_GRID = [0.1, 0.3, 1.0, 3.0]
REAL_FLAG_RATE = 0.05


def load():
    X = np.load(F / "feats.npy").astype(np.float32)
    X /= np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8
    L = np.load(F / "crop_labels.npy")
    idx = pd.read_csv(F / "index.csv", dtype=str, keep_default_na=False)
    idx["label"] = idx["label"].astype(int)
    return X, L, idx


def select(idx, split, gens):
    return ((idx.split == split) &
            ((idx.label == 0) | ((idx.label == 1) & idx.generator.isin(gens)))).values


def fit(X, L, mask, C, kinds=None):
    Xc, yc = X[mask].reshape(-1, X.shape[-1]), L[mask].reshape(-1)
    keep = yc >= 0
    if kinds is None:
        return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xc[keep], yc[keep])
    kc = np.repeat(kinds[mask], X.shape[1])[keep]
    yk = yc[keep]
    w = np.ones(len(yk))
    base, cardd, ai = (kc == "base_original"), (kc == "real_damaged"), (yk == 1)
    if base.any() and cardd.any():
        w[base] = cardd.sum() / base.sum()            # untouched web photos count as much as all CarDD
    if ai.any():
        w[ai] = w[~ai].sum() / ai.sum()               # AI crops count as much as all real crops
    return LogisticRegression(C=C, max_iter=3000).fit(Xc[keep], yk, sample_weight=w)


def image_scores(clf, X):
    n = X.shape[0]
    return clf.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(n, 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-balance", action="store_true")
    args = ap.parse_args()
    X, L, idx = load()
    MODELS.mkdir(exist_ok=True)
    kinds = idx.kind.values if args.domain_balance else None
    summary = {}
    for name, gens in EXPERIMENTS.items():
        tr, va = select(idx, "train", gens), select(idx, "val", gens)
        if (idx.label.values[tr] == 1).sum() == 0 or (idx.label.values[va] == 1).sum() == 0:
            print(f"{name}: no AI images from {gens} in train or val yet, skipped")
            continue
        y_va = idx.label.values[va]
        best = None
        for C in C_GRID:
            clf = fit(X, L, tr, C, kinds)
            auc = roc_auc_score(y_va, image_scores(clf, X[va]).max(1))
            print(f"  {name} C={C}: val ROC-AUC {auc:.3f}")
            if best is None or auc > best[0]:
                best = (auc, C, clf)
        auc, C, clf = best
        s_va = image_scores(clf, X[va]).max(1)
        cal = LogisticRegression().fit(s_va.reshape(-1, 1), y_va)
        p_va = cal.predict_proba(s_va.reshape(-1, 1))[:, 1]
        ref = (y_va == 0) & (idx.kind.values[va] == "base_original") if args.domain_balance else (y_va == 0)
        thr = float(np.quantile(p_va[ref], 1 - REAL_FLAG_RATE))
        flagged = p_va > thr
        kinds = idx.kind.values[va]
        res = {"gens": gens, "C": C, "val_auc": round(auc, 4), "threshold": round(thr, 4),
               "domain_balance": bool(args.domain_balance),
               "val_false_flag_real": round(float(flagged[y_va == 0].mean()), 4)}
        for k in ("base_original", "real_damaged"):
            sel = (y_va == 0) & (idx.kind.values[va] == k)
            if sel.any():
                res[f"val_false_flag_{k}"] = round(float(flagged[sel].mean()), 4)
        for k in ("paste_back", "whole_edit", "full_gen"):
            sel = (y_va == 1) & (kinds == k)
            if sel.any():
                res[f"val_recall_{k}"] = round(float(flagged[sel].mean()), 4)
        out = MODELS / f"{name}.joblib"
        if args.domain_balance and out.exists() and not (MODELS / f"{name}_v1.joblib").exists():
            shutil.copy(out, MODELS / f"{name}_v1.joblib")            # keep the first model
        joblib.dump({"clf": clf, "cal": cal, "threshold": thr, **res}, out)
        summary[name] = res
        print(f"{name}: {res}")
    (MODELS / "val_summary.json").write_text(json.dumps(summary, indent=1))
    print("saved models/*.joblib and models/val_summary.json")


if __name__ == "__main__":
    main()