"""Build the 150-post AITA pilot set from the Scruples Anecdotes corpus.

Source: Allen AI Scruples v1.0 anecdotes release (Lourie, Le Bras & Choi 2020,
arXiv:2008.09094), downloaded from
https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz
and extracted into data/raw/scruples/anecdotes/. All three splits
(train/dev/test) are pooled — this pilot is for measuring judge geometry, not
for training, so the corpus split is irrelevant.

Per post we form binary verdict counts mirroring Paper A's YTA/NTA binarization
(and Scruples' own binarized_label_scores):
    n_unacceptable = AUTHOR + EVERYBODY   (author is in the wrong)
    n_acceptable   = OTHER + NOBODY       (author is not in the wrong)
    p_unacceptable = n_unacceptable / (n_unacceptable + n_acceptable)
INFO votes are recorded but excluded from the binary denominator.

Filters: n_binary >= 40; text length in [400, 4000] chars; text non-empty and
not a [removed]/[deleted] stub.

Selection: stratify by consensus = max(p, 1-p) into five buckets
[0.5,0.6), [0.6,0.7), [0.7,0.8), [0.8,0.9), [0.9,1.0]; sample 30 per bucket
with numpy seed 20260709. Any bucket with fewer than 30 qualifying posts is
taken whole and topped up from the nearest bucket(s) by bucket distance.

Output: data/aita_pilot_150.csv. Run from the repo root:
    .venv/bin/python pipeline/build_aita_pilot.py
"""

from __future__ import annotations

import glob
import html
import json
import os

import numpy as np
import pandas as pd

SEED = 20260709
RAW_GLOB = "data/raw/scruples/anecdotes/*.scruples-anecdotes.jsonl"
OUT_CSV = "data/aita_pilot_150.csv"
MIN_BINARY = 40
MIN_CHARS, MAX_CHARS = 400, 4000
PER_BUCKET = 30

BUCKETS = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]


def consensus_bucket(c: float) -> str:
    """Map consensus in [0.5, 1.0] to one of the five bucket labels."""
    if c >= 0.9:
        return "0.9-1.0"
    idx = int((c - 0.5) / 0.1)  # 0..3 for [0.5,0.9)
    return BUCKETS[idx]


def load_pool() -> pd.DataFrame:
    rows = []
    paths = sorted(glob.glob(RAW_GLOB))
    assert paths, f"no raw files matching {RAW_GLOB}; download the corpus first"
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                ls = r["label_scores"]
                text = html.unescape(r.get("text") or "")  # undo reddit &amp; etc.
                rows.append(
                    dict(
                        post_id=r["post_id"],
                        title=html.unescape(r.get("title") or ""),
                        text=text,
                        n_author=ls["AUTHOR"],
                        n_other=ls["OTHER"],
                        n_everybody=ls["EVERYBODY"],
                        n_nobody=ls["NOBODY"],
                        n_info=ls["INFO"],
                    )
                )
    df = pd.DataFrame(rows)
    # A post_id should be unique across splits; keep first if not.
    df = df.drop_duplicates(subset="post_id", keep="first").reset_index(drop=True)

    df["n_unacceptable"] = df["n_author"] + df["n_everybody"]
    df["n_acceptable"] = df["n_other"] + df["n_nobody"]
    df["n_binary"] = df["n_unacceptable"] + df["n_acceptable"]
    df["text_len"] = df["text"].str.len()
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    stripped = df["text"].str.strip()
    low = stripped.str.lower()
    ok = (
        (df["n_binary"] >= MIN_BINARY)
        & df["text_len"].between(MIN_CHARS, MAX_CHARS)
        & (stripped.str.len() > 0)
        & ~low.str.startswith("[removed]")
        & ~low.str.startswith("[deleted]")
    )
    out = df[ok].copy()
    out["p_unacceptable"] = out["n_unacceptable"] / out["n_binary"]
    out["consensus"] = np.maximum(out["p_unacceptable"], 1 - out["p_unacceptable"])
    out["bucket"] = out["consensus"].map(consensus_bucket)
    return out


def stratified_sample(pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    pool = pool.sort_values("post_id").reset_index(drop=True)  # deterministic base order

    taken_ids: list[str] = []
    shortfall: dict[str, int] = {}
    for b in BUCKETS:
        cand = pool[pool["bucket"] == b]
        if len(cand) <= PER_BUCKET:
            taken_ids += list(cand["post_id"])
            shortfall[b] = PER_BUCKET - len(cand)
        else:
            idx = rng.choice(len(cand), size=PER_BUCKET, replace=False)
            taken_ids += list(cand.iloc[idx]["post_id"])
            shortfall[b] = 0

    # Top up any short bucket from the nearest bucket(s) by index distance.
    for b, miss in shortfall.items():
        if miss <= 0:
            continue
        bi = BUCKETS.index(b)
        for dist in range(1, len(BUCKETS)):
            if miss == 0:
                break
            for nb_idx in (bi - dist, bi + dist):
                if miss == 0 or not (0 <= nb_idx < len(BUCKETS)):
                    continue
                cand = pool[(pool["bucket"] == BUCKETS[nb_idx]) & ~pool["post_id"].isin(taken_ids)]
                take = min(miss, len(cand))
                if take > 0:
                    idx = rng.choice(len(cand), size=take, replace=False)
                    taken_ids += list(cand.iloc[idx]["post_id"])
                    miss -= take
        shortfall[b] = miss  # any residual means the whole pool ran dry

    sel = pool[pool["post_id"].isin(taken_ids)].copy()
    sel = sel.sort_values(["bucket", "consensus", "post_id"]).reset_index(drop=True)
    return sel


def main() -> None:
    df = load_pool()
    print(f"pool: {len(df)} unique posts loaded from raw jsonl")
    pool = apply_filters(df)
    print(
        f"after filters (n_binary>={MIN_BINARY}, {MIN_CHARS}<=len<={MAX_CHARS}, "
        f"non-deleted): {len(pool)} posts"
    )
    print("qualifying pool per bucket:")
    print(pool["bucket"].value_counts().reindex(BUCKETS).to_string())

    sel = stratified_sample(pool)
    cols = [
        "post_id", "title", "text",
        "n_author", "n_other", "n_everybody", "n_nobody", "n_info",
        "n_binary", "p_unacceptable", "consensus", "bucket",
    ]
    out = sel[cols]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")
    print("selected per bucket:")
    print(out["bucket"].value_counts().reindex(BUCKETS).to_string())
    nb = out["n_binary"]
    print(
        f"n_binary: min={nb.min()} median={nb.median():.0f} max={nb.max()}  |  "
        f"mean text len={out['text'].str.len().mean():.0f} chars"
    )


if __name__ == "__main__":
    main()
