"""Turn A's test predictions into the Markdown tables for the write-up.
    python eval_report.py test_all.csv > results.md

Input: predictions/test_all.csv from A's 12_test.py (one row per test image per experiment and
variant, with label, kind, generator, lighting, color, mask_frac, prob, flagged).
Every rate comes with its count n and a 95% Wilson interval; groups under 30 are marked.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

NAMES = {"sd15": "SD 1.5", "kontext": "FLUX.1 Kontext", "flux": "FLUX.1 Fill", "sdxl": "SDXL (held out)", "real": "real"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(df):
    n = len(df)
    if n == 0:
        return "n=0"
    k = int(df.flagged.sum())
    lo, hi = wilson(k, n)
    return f"{k / n:.0%} [{lo:.0%}–{hi:.0%}], n={n}" + (" (too small)" if n < 30 else "")


def table(title, groups, what="Flagged"):
    print(f"\n### {title}\n\n| Group | {what} [95% CI] |\n| --- | --- |")
    for name, g in groups:
        print(f"| {name} | {rate(g)} |")


def auc(real, ai):
    both = pd.concat([real, ai])
    return f"{roc_auc_score(both.label, both.prob):.3f}" if both.label.nunique() == 2 else "–"


p = pd.read_csv(sys.argv[1], keep_default_na=False)
p["flagged"] = p.flagged.astype(str).str.lower().isin(["true", "1"])
p["label"] = p.label.astype(int)
p["prob"] = p.prob.astype(float)

for exp, e in p.groupby("experiment", sort=False):
    print(f"\n## Experiment: {exp}")
    c = e[e.variant == "clean"]
    real, ai = c[c.label == 0], c[c.label == 1]
    print("\n### Headline\n\n| Test set | Flagged [95% CI] | ROC-AUC vs all real photos |\n| --- | --- | --- |")
    for (k, g), d in ai.groupby(["kind", "generator"]):
        print(f"| {k} · {NAMES.get(g, g)} | {rate(d)} | {auc(real, d)} |")
    for k, d in real.groupby("kind"):
        print(f"| {k} (real, false flags) | {rate(d)} | |")
    rt = c[c.kind == "round_trip"]
    if len(rt):
        print(f"| round trip (no edit, diagnostic) | {rate(rt)} | |")

    pb = ai[ai.kind == "paste_back"].copy()
    if len(pb) and (pb.mask_frac != "").any():
        pb = pb[pb.mask_frac != ""]
        pb["size"] = pd.cut(pb.mask_frac.astype(float), [0, 0.06, 0.10, 1.0],
                            labels=["3–6% of image", "6–10%", "10–15%"])
        table("Paste-back recall by edit size", [(str(s), d) for s, d in pb.groupby("size", observed=True)])
    for col in ("lighting", "color"):
        table(f"Paste-back recall by {col}", [(s or "unknown", d) for s, d in pb.groupby(col)])
        table(f"False flags on real photos by {col}", [(s or "unknown", d) for s, d in real.groupby(col)])

    v = e[e.variant != "clean"]
    if len(v):
        print("\n### Robustness: re-saved or resized test photos\n")
        print("| Variant | Paste-back recall | False flags on real photos |\n| --- | --- | --- |")
        for name, d in [("original", c)] + list(v.groupby("variant")):
            print(f"| {name} | {rate(d[d.kind == 'paste_back'])} | {rate(d[d.label == 0])} |")
