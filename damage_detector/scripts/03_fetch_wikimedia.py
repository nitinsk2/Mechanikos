"""Step 4b: pull freely licensed photos of undamaged cars from Wikimedia Commons.
Run from the project root:
    python scripts/03_fetch_wikimedia.py --per-query 80 --max-total 1000
Downloads 1280-px-wide versions into raw/wikimedia/ and writes manifests/wikimedia.csv
(title, page URL, license, author) for every file. Then REVIEW the folder by eye and delete
rejects; later steps only use files that still exist.

Only CC BY, CC BY-SA, CC0 and public-domain files are kept (no NC / ND: we make derivatives).
"""
import argparse
import csv
import html
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks for a descriptive User-Agent with a way to contact you. Put your own
# contact (email or GitHub URL) in place of the placeholder before running.
UA = "car-damage-hackathon-dataset/0.1 (contact: https://github.com/<Dope-Ahmad>)"

QUERIES = [
    # body types and settings
    "parked car", "car parking lot", "sedan side view", "hatchback parked",
    "SUV parked street", "pickup truck parked", "minivan parked", "station wagon parked",
    "compact car street", "crossover SUV parked", "coupe parked", "car driveway",
    # the six colours the slice report uses
    "white car parked", "silver car parked", "black car parked", "red car parked",
    "blue car parked", "grey car parked",
    # lighting variety (Commons skews to sunny daylight)
    "car parked overcast", "car parked dusk", "car parked night", "car parking garage",
    "car parked rain", "car parked snow",
]
EXCLUDE = "-crash -wreck -damaged -accident -burnt -rust -toy -diecast -miniature -interior" \
          "-engine -drawing -logo -racing -rally -museum"
OK_LICENSE = re.compile(r"^(cc[ -]by(-sa)?( \d\.\d)?|cc0|public domain|pd)", re.I)
BAD_LICENSE = re.compile(r"\b(nc|nd)\b", re.I)

OUT = Path("raw/wikimedia")
MANIFEST = Path("manifests/wikimedia.csv")


def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def search(session, query, limit):
    params = dict(action="query", format="json", formatversion=2, generator="search",
                  gsrsearch=f"{query} filetype:bitmap {EXCLUDE}", gsrnamespace=6, gsrlimit=50,
                  prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth=1280,
                  iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist")
    got = []
    while len(got) < limit:
        r = session.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        got += data.get("query", {}).get("pages", [])
        if "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(0.5)
    return got[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=80)
    ap.add_argument("--max-total", type=int, default=1000)
    args = ap.parse_args()
    if "REPLACE_WITH" in UA:
        raise SystemExit("Edit UA at the top of this script with your own contact first.")

    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA

    rows, seen = [], set()
    if MANIFEST.exists():  # resume
        with open(MANIFEST, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        seen = {r["pageid"] for r in rows}

    for q in QUERIES:
        if len(rows) >= args.max_total:
            break
        added = 0
        for page in search(s, q, args.per_query):
            pid = str(page.get("pageid"))
            if pid in seen or "imageinfo" not in page:
                continue
            seen.add(pid)
            ii = page["imageinfo"][0]
            meta = ii.get("extmetadata", {})
            lic = strip_html(meta.get("LicenseShortName", {}).get("value"))
            if not OK_LICENSE.match(lic) or BAD_LICENSE.search(lic):
                continue
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if min(ii.get("width", 0), ii.get("height", 0)) < 900:
                continue
            url = ii.get("thumburl") or ii["url"]
            ext = ".png" if ii["mime"] == "image/png" else ".jpg"
            fname = f"wm_{pid}{ext}"
            try:
                r = s.get(url, timeout=60)
                r.raise_for_status()
            except requests.RequestException as e:
                print("   skip", fname, e)
                time.sleep(5)
                continue
            (OUT / fname).write_bytes(r.content)
            rows.append(dict(file=fname, pageid=pid, title=page["title"],
                             page_url=ii.get("descriptionurl", ""), license=lic,
                             license_url=strip_html(meta.get("LicenseUrl", {}).get("value")),
                             author=strip_html(meta.get("Artist", {}).get("value"))[:200],
                             query=q))
            added += 1
            time.sleep(1.0)  # be polite
        print(f"{q!r}: +{added} (total {len(rows)})")
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["file", "pageid", "title", "page_url", "license",
                                               "license_url", "author", "query"])
            wr.writeheader()
            wr.writerows(rows)

    print(f"\n{len(rows)} files in {OUT}/ - now review them by eye and delete rejects.")


if __name__ == "__main__":
    main()
