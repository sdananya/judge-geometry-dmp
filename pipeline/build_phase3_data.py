"""Phase 3 data build (README pre-registration, 2026-07-14). Seed 20260714 throughout.

Stages (run: .venv/bin/python pipeline/build_phase3_data.py <stage>):
  pilot    -> data/aita_pilot_500.csv        (150 existing + 70/bucket top-up)
  rewrites -> results/phase3/rewrites.jsonl  (150 copied verbatim from phase2 + 350 new)
              + results/phase3/rewrite_gate.json
  topics   -> data/topics_500.json           (haiku, 8 fixed labels, rewrite as input)
  g0topics -> data/g0_topics.json            (per-topic value priors from mattboraske)
  fewshot  -> data/fewshot_pool.json         (8 elicitation examples, non-overlapping)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from build_aita_pilot import BUCKETS, apply_filters, load_pool
from run_aita_pilot import MEM_SYS, MEM_USER
from run_phase2 import REWRITE_SYS_V2

SEED = 20260714
PER_BUCKET = 100
CSV_150 = ROOT / "data" / "aita_pilot_150.csv"
CSV_500 = ROOT / "data" / "aita_pilot_500.csv"
P2_REWRITES = ROOT / "results" / "phase2" / "rewrites.jsonl"
P3 = ROOT / "results" / "phase3"
COLS = ["post_id", "title", "text", "n_author", "n_other", "n_everybody", "n_nobody",
        "n_info", "n_binary", "p_unacceptable", "consensus", "bucket"]

TOPICS = ["romantic_relationships", "family", "money_property", "work",
          "friendship_social", "events_celebrations", "household_neighbors", "other"]

TOPIC_SYS = (
    "You label short interpersonal-dilemma stories with exactly one topic. "
    "The allowed labels are:\n"
    "romantic_relationships (partners, dating, exes, marriage-as-couple)\n"
    "family (parents, children, siblings, in-laws, relatives)\n"
    "money_property (debts, loans, rent, purchases, inheritance, damage, ownership)\n"
    "work (jobs, coworkers, bosses, customers, hiring)\n"
    "friendship_social (friends, roommates-as-friends, social groups)\n"
    "events_celebrations (weddings, parties, holidays, funerals, trips)\n"
    "household_neighbors (neighbors, shared housing logistics, pets at home, chores)\n"
    "other (anything that fits none of the above)\n"
    "Pick the single topic the central conflict is about. Answer with exactly one "
    "label, verbatim, lowercase, and nothing else.")


def load_items_500() -> list[dict]:
    return list(csv.DictReader(open(CSV_500, encoding="utf-8")))


# ---------------- stage 1: 500-item pilot set ----------------

def pilot():
    old = list(csv.DictReader(open(CSV_150, encoding="utf-8")))
    old_ids = [r["post_id"] for r in old]
    assert len(old_ids) == 150 and len(set(old_ids)) == 150
    old_by_bucket = Counter(r["bucket"] for r in old)

    pool = apply_filters(load_pool())
    pool = pool.sort_values("post_id").reset_index(drop=True)  # deterministic base order
    print(f"qualifying pool: {len(pool)}")

    rng = np.random.default_rng(SEED)
    taken = list(old_ids)
    for b in BUCKETS:
        need = PER_BUCKET - old_by_bucket[b]
        cand = pool[(pool["bucket"] == b) & ~pool["post_id"].isin(taken)]
        assert len(cand) >= need, f"bucket {b}: only {len(cand)} candidates for {need}"
        idx = rng.choice(len(cand), size=need, replace=False)
        taken += list(cand.iloc[idx]["post_id"])
        print(f"  {b}: had {old_by_bucket[b]}, +{need} from {len(cand)} candidates")

    sel = pool[pool["post_id"].isin(taken)].copy()
    sel = sel.sort_values(["bucket", "consensus", "post_id"]).reset_index(drop=True)
    sel[COLS].to_csv(CSV_500, index=False)

    # verify
    out = list(csv.DictReader(open(CSV_500, encoding="utf-8")))
    assert len(out) == 500, len(out)
    bc = Counter(r["bucket"] for r in out)
    assert all(bc[b] == PER_BUCKET for b in BUCKETS), bc
    new_ids = {r["post_id"] for r in out}
    assert set(old_ids) <= new_ids, "missing old post_ids"
    # old rows must carry identical data (same pool, same computation)
    out_by = {r["post_id"]: r for r in out}
    for r in old:
        n = out_by[r["post_id"]]
        assert r["text"] == n["text"] and r["bucket"] == n["bucket"]
        assert abs(float(r["p_unacceptable"]) - float(n["p_unacceptable"])) < 1e-12
    print(f"wrote {CSV_500}: 500 rows, {dict(bc)}, all 150 old ids present & identical")


# ---------------- stage 2: rewrites + gate ----------------

FAB_AGE_RE = re.compile(r"\d{2}[- ]year[- ]old")
ORIG_DEMO_RE = re.compile(
    r"\b\d{2}\s*[MFmf]\b|\b[MFmf]\s*\d{2}\b|\byear[- ]old\b|\(\d{2}[MFmf]?\)")


def rewrites():
    from llm import map_calls
    P3.mkdir(parents=True, exist_ok=True)
    items = load_items_500()
    old_lines = {}  # post_id -> raw jsonl line, copied verbatim (no re-generation)
    for line in open(P2_REWRITES):
        old_lines[json.loads(line)["post_id"]] = line.rstrip("\n")
    new_items = [it for it in items if it["post_id"] not in old_lines]
    print(f"{len(old_lines)} copied from phase2, {len(new_items)} to generate")

    jobs = [{"provider": "openai", "model": "gpt-4o-mini", "system": REWRITE_SYS_V2,
             "user": f"TITLE: {it['title']}\n\n{it['text']}", "temperature": 0.3,
             "max_tokens": 500} for it in new_items]
    rw = map_calls(jobs, concurrency=16)
    assert all(r for r in rw), "some rewrite calls failed"

    mem_jobs, mem_meta = [], []
    for i, r in enumerate(rw):
        for s in range(3):
            mem_jobs.append({"provider": "openai", "model": "gpt-4o-mini",
                             "system": MEM_SYS, "user": MEM_USER.format(text=r or ""),
                             "temperature": 1.0, "max_tokens": 4, "seed": s})
            mem_meta.append(i)
    mem = map_calls(mem_jobs, concurrency=16)
    flags = {}
    for i, ans in zip(mem_meta, mem):
        flags.setdefault(i, []).append((ans or "").strip().upper().startswith("YES"))

    new_rows = {}
    for i, it in enumerate(new_items):
        new_rows[it["post_id"]] = json.dumps({
            "post_id": it["post_id"], "rewrite": rw[i],
            "anchored": "main actor" in (rw[i] or "").lower(),
            "mem_flagged": any(flags.get(i, [True]))})

    with open(P3 / "rewrites.jsonl", "w") as f:
        for it in items:  # csv order
            f.write((old_lines.get(it["post_id"]) or new_rows[it["post_id"]]) + "\n")

    rows = {json.loads(l)["post_id"]: json.loads(l) for l in open(P3 / "rewrites.jsonl")}
    assert len(rows) == 500
    n_anch = sum(rows[it["post_id"]]["anchored"] for it in items)
    n_mem = sum(rows[it["post_id"]]["mem_flagged"] for it in items)
    n_fab = 0
    for it in items:
        r = rows[it["post_id"]]["rewrite"] or ""
        if FAB_AGE_RE.search(r) and not ORIG_DEMO_RE.search(it["title"] + " " + it["text"]):
            n_fab += 1
    gate = {"anchored_frac": n_anch / 500, "gate_ok": n_anch / 500 >= 0.95,
            "mem_flagged": n_mem, "fabricated_demo_frac": n_fab / 500}
    (P3 / "rewrite_gate.json").write_text(json.dumps(gate))
    print(json.dumps(gate))


# ---------------- topic labeling (shared) ----------------

def _parse_topic(text: str | None) -> str | None:
    t = (text or "").strip().strip(".\"' ").lower()
    return t if t in TOPICS else None


def label_topics(texts: list[str], concurrency: int = 48) -> tuple[list[str], int, int]:
    """Haiku, temp 0, one call per text; retry once on parse failure, then 'other'."""
    from llm import map_calls

    def jobs(idxs, seed):
        return [{"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
                 "system": TOPIC_SYS, "user": texts[i], "temperature": 0.0,
                 "max_tokens": 16, "seed": seed} for i in idxs]

    out: list[str | None] = [None] * len(texts)
    first = map_calls(jobs(range(len(texts)), 0), concurrency=concurrency)
    for i, t in enumerate(first):
        out[i] = _parse_topic(t)
    retry_idx = [i for i, t in enumerate(out) if t is None]
    if retry_idx:
        second = map_calls(jobs(retry_idx, 1), concurrency=concurrency)
        for i, t in zip(retry_idx, second):
            out[i] = _parse_topic(t)
    n_fallback = sum(t is None for t in out)
    return [t or "other" for t in out], len(retry_idx), n_fallback


# ---------------- stage 3: topics for the 500 ----------------

def topics():
    rows = [json.loads(l) for l in open(P3 / "rewrites.jsonl")]
    assert len(rows) == 500
    labels, n_retry, n_fb = label_topics([r["rewrite"] or "" for r in rows])
    out = {r["post_id"]: t for r, t in zip(rows, labels)}
    (ROOT / "data" / "topics_500.json").write_text(json.dumps(out, indent=1))
    dist = Counter(labels)
    print(f"retried: {n_retry}, parse-fallback-to-other: {n_fb}")
    print(json.dumps({t: dist.get(t, 0) for t in TOPICS}, indent=1))


# ---------------- stage 4: per-topic G0 ----------------

def g0topics():
    import pandas as pd
    from run_aita_pilot import load_values

    values = load_values()
    assert len(values) == 59
    mentions = [json.loads(l) for l in open(ROOT / "data/raw/mattboraske/g0_value_mentions.jsonl")]
    sample = [json.loads(l) for l in open(ROOT / "data/raw/mattboraske/g0_sample.jsonl")]
    # join key = (post_idx, comment_slot); mention rows carry post_idx already — verify
    skeys = {(r["post_idx"], r["comment_slot"]) for r in sample}
    assert all((m["post_idx"], m["comment_slot"]) in skeys for m in mentions)
    post_idxs = sorted({m["post_idx"] for m in mentions})
    print(f"{len(post_idxs)} distinct source posts")

    df = pd.read_parquet(ROOT / "data/raw/mattboraske/test-00000-of-00001.parquet")
    texts = []
    for pi in post_idxs:
        row = df.iloc[pi]
        title = row["submission_title"] if isinstance(row["submission_title"], str) else ""
        body = row["submission_text"] if isinstance(row["submission_text"], str) else ""
        texts.append(f"TITLE: {title}\n\n{body}"[:1500])
    labels, n_retry, n_fb = label_topics(texts)
    topic_of = dict(zip(post_idxs, labels))
    print(f"retried: {n_retry}, parse-fallback-to-other: {n_fb}")

    def smoothed(counter: Counter) -> dict:
        arr = np.array([counter.get(v, 0) for v in values], dtype=float) + 1.0
        arr /= arr.sum()
        return {v: float(p) for v, p in zip(values, arr)}

    global_counts = Counter()
    topic_counts: dict[str, Counter] = {t: Counter() for t in TOPICS}
    for m in mentions:
        global_counts.update(m["values"])
        topic_counts[topic_of[m["post_idx"]]].update(m["values"])

    g_global = smoothed(global_counts)
    out_topics, fellback, per_topic_n = {}, [], {}
    for t in TOPICS:
        n = sum(topic_counts[t].values())
        per_topic_n[t] = n
        if n < 100:
            out_topics[t] = dict(g_global)
            fellback.append(t)
        else:
            out_topics[t] = smoothed(topic_counts[t])
        assert list(out_topics[t]) == values and abs(sum(out_topics[t].values()) - 1) < 1e-9

    out = {"global": g_global, "topics": out_topics,
           "meta": {"seed": SEED, "n_posts": len(post_idxs),
                    "n_mentions": int(sum(global_counts.values())),
                    "mentions_per_topic": per_topic_n,
                    "fallback_topics_lt100": fellback,
                    "post_topic_counts": dict(Counter(labels)),
                    "smoothing": "add-one over 59 values",
                    "labeler": "claude-haiku-4-5-20251001 temp0, title+text[:1500]"}}
    (ROOT / "data" / "g0_topics.json").write_text(json.dumps(out, indent=1))
    print("mentions per topic:", json.dumps(per_topic_n, indent=1))
    print("fell back to global:", fellback)


# ---------------- stage 5: few-shot pool ----------------

def fewshot():
    from llm import map_calls
    used = {r["post_id"] for r in load_items_500()}
    pool = apply_filters(load_pool())
    pool = pool[~pool["post_id"].isin(used)].sort_values("post_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    # (bucket, majority-filter, count): 1/1/1 low buckets, 2 from 0.8-0.9,
    # 3 from 0.9-1.0 with a mix of majorities (2 acceptable-majority + 1 unacceptable).
    plan = [("0.5-0.6", None, 1), ("0.6-0.7", None, 1), ("0.7-0.8", None, 1),
            ("0.8-0.9", None, 2), ("0.9-1.0", "acc", 2), ("0.9-1.0", "unacc", 1)]
    picked = []
    for b, maj, k in plan:
        cand = pool[pool["bucket"] == b]
        if maj == "acc":
            cand = cand[cand["p_unacceptable"] < 0.5]
        elif maj == "unacc":
            cand = cand[cand["p_unacceptable"] > 0.5]
        cand = cand[~cand["post_id"].isin([p["post_id"] for p in picked])]
        # draw k + backups in one deterministic ordering; use first k that pass anchor
        idx = rng.choice(len(cand), size=min(len(cand), k + 6), replace=False)
        queue = cand.iloc[idx].to_dict("records")
        taken = 0
        for it in queue:
            if taken == k:
                break
            rw = map_calls([{"provider": "openai", "model": "gpt-4o-mini",
                             "system": REWRITE_SYS_V2,
                             "user": f"TITLE: {it['title']}\n\n{it['text']}",
                             "temperature": 0.3, "max_tokens": 500}])[0]
            if rw and "main actor" in rw.lower():
                picked.append({"post_id": it["post_id"], "bucket": b,
                               "p_unacceptable": float(it["p_unacceptable"]),
                               "rewrite": rw,
                               "pct_unacceptable": int(round(100 * float(it["p_unacceptable"])))})
                taken += 1
            else:
                print(f"  anchor FAIL for {it['post_id']} in {b}, trying next candidate")
        assert taken == k, f"could not fill {b}/{maj}"

    assert len(picked) == 8 and len({p["post_id"] for p in picked}) == 8
    assert all("main actor" in p["rewrite"].lower() for p in picked)
    assert not any(p["post_id"] in used for p in picked)
    (ROOT / "data" / "fewshot_pool.json").write_text(json.dumps(picked, indent=1))
    print(f"wrote data/fewshot_pool.json: "
          f"{[(p['bucket'], p['pct_unacceptable']) for p in picked]}")


if __name__ == "__main__":
    {"pilot": pilot, "rewrites": rewrites, "topics": topics,
     "g0topics": g0topics, "fewshot": fewshot}[sys.argv[1]]()
