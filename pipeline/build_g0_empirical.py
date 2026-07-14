"""Build empirical base measure G0 over the 59 taxonomy values from real human AITA rationales.

Mirrors Paper A (Russo et al., EACL 2026): G0(v) = frequency of value v across human
rationales, here estimated from top-level YTA/NTA comments in
MattBoraske/reddit-AITA-submissions-and-comments-multiclass (test split parquet,
downloaded to data/raw/mattboraske/).

Stages (run via: .venv/bin/python pipeline/build_g0_empirical.py <stage>):
  sample   -> data/raw/mattboraske/g0_sample.jsonl (+ prints 10 raw comments to read)
  extract  -> data/raw/mattboraske/g0_value_mentions.jsonl (haiku value extraction)
  aggregate-> data/g0_empirical.json + data/G0_NOTE.md (add-one smoothing)
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mattboraske"
PARQUET = RAW / "test-00000-of-00001.parquet"
SAMPLE_FILE = RAW / "g0_sample.jsonl"
MENTIONS_FILE = RAW / "g0_value_mentions.jsonl"
G0_FILE = ROOT / "data" / "g0_empirical.json"
NOTE_FILE = ROOT / "data" / "G0_NOTE.md"

SEED = 20260714
N_TARGET_COMMENTS = 1500
MAX_PER_POST = 5
LEN_MIN, LEN_MAX = 80, 1200
KEEP_CLASSES = {"YTA", "NTA"}

# Bot/meta junk filters (AutoModerator copypasta, bot signatures, mod messages)
BOT_PATTERNS = [
    r"I am a bot", r"AutoModerator", r"\bthis bot\b", r"performed automatically",
    r"contact the moderators", r"AUTOMOD", r"Judgement Bot", r"\^\^\^",
    r"Welcome to /?r/AmITheAsshole", r"INFO is not a judgement",
]
BOT_RE = re.compile("|".join(BOT_PATTERNS), re.I)


def load_values() -> list[str]:
    txt = (ROOT / "pipeline" / "values_taxonomy.yaml").read_text()
    return [m.group(1).strip() for m in re.finditer(r"^  - (.+)$", txt, re.M)]


def sample():
    import pandas as pd

    df = pd.read_parquet(PARQUET)
    print(f"{len(df)} posts in parquet")
    rng = random.Random(SEED)
    order = list(range(len(df)))
    rng.shuffle(order)

    rows, n_posts, n_bot_dropped = [], 0, 0
    for idx in order:
        if len(rows) >= N_TARGET_COMMENTS:
            break
        post = df.iloc[idx]
        taken = 0
        for k in range(1, 11):
            if taken >= MAX_PER_POST or len(rows) >= N_TARGET_COMMENTS:
                break
            c = post[f"top_comment_{k}"]
            cls = post[f"top_comment_{k}_classification"]
            if not isinstance(c, str) or not isinstance(cls, str):
                continue
            if cls.strip().upper() not in KEEP_CLASSES:
                continue
            c = c.strip()
            if not (LEN_MIN <= len(c) <= LEN_MAX):
                continue
            if BOT_RE.search(c):
                n_bot_dropped += 1
                continue
            rows.append({"post_idx": int(idx), "comment_slot": k,
                         "classification": cls.strip().upper(), "text": c})
            taken += 1
        if taken:
            n_posts += 1

    with open(SAMPLE_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cls_counts = Counter(r["classification"] for r in rows)
    print(f"sampled {len(rows)} comments from {n_posts} posts "
          f"(bot-dropped {n_bot_dropped}); classes {dict(cls_counts)}")
    print(f"wrote {SAMPLE_FILE}")

    # print 10 raw comments for manual reading (fixed sub-seed)
    view = random.Random(SEED + 1).sample(rows, 10)
    for i, r in enumerate(view):
        print(f"\n===== READ {i+1} [{r['classification']}] (post {r['post_idx']}, "
              f"slot {r['comment_slot']}, {len(r['text'])} ch) =====\n{r['text']}")


VALUES_SYS = """You classify which moral values a rationale invokes. You are given a fixed list of values and a rationale. Return a JSON array (no code fences) of the values from the list that the rationale clearly relies on, most central first, at most 3. Use EXACT spellings from the list. If none apply, return [].
VALUES: {values}"""


def extract():
    sys.path.insert(0, str(ROOT / "pipeline"))
    from llm import map_calls

    values = load_values()
    vset = set(values)
    rows = [json.loads(l) for l in open(SAMPLE_FILE)]
    jobs = [{"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
             "system": VALUES_SYS.format(values=", ".join(values)),
             "user": r["text"][:1500], "temperature": 0.0, "max_tokens": 100}
            for r in rows]
    print(f"{len(jobs)} extraction calls")
    texts = map_calls(jobs, concurrency=48)
    n_fail = sum(t is None for t in texts)
    with open(MENTIONS_FILE, "w") as f:
        for r, t in zip(rows, texts):
            try:
                s = t or "[]"
                vals = [v for v in json.loads(s[s.find("["):s.rfind("]") + 1]) if v in vset]
            except Exception:
                vals = []
            f.write(json.dumps({"post_idx": r["post_idx"], "comment_slot": r["comment_slot"],
                                "classification": r["classification"], "values": vals}) + "\n")
    print(f"wrote {MENTIONS_FILE} ({n_fail} failed calls)")


def aggregate():
    sys.path.insert(0, str(ROOT / "metrics"))
    import numpy as np
    from geometry import normalized_entropy, top_k_mass

    values = load_values()
    assert len(values) == 59
    rows = [json.loads(l) for l in open(MENTIONS_FILE)]
    counts = Counter()
    for r in rows:
        counts.update(r["values"])
    unknown = set(counts) - set(values)
    assert not unknown, f"non-taxonomy values leaked: {unknown}"

    n_mentions = sum(counts.values())
    n_empty = sum(1 for r in rows if not r["values"])
    raw = np.array([counts.get(v, 0) for v in values], dtype=float)
    zero_vals = [v for v in values if counts.get(v, 0) == 0]

    smoothed = raw + 1.0  # add-one smoothing over mention counts
    g0 = smoothed / smoothed.sum()
    assert abs(g0.sum() - 1.0) < 1e-9

    top = sorted(zip(values, raw / raw.sum(), raw), key=lambda t: -t[1])[:10]
    t10 = top_k_mass(raw, 10)
    ent = normalized_entropy(g0)
    ent_raw = normalized_entropy(raw)

    out = {"source": "MattBoraske/reddit-AITA-submissions-and-comments-multiclass "
                     "(HF, test split), top-level YTA/NTA comments, values extracted by "
                     "claude-haiku-4-5-20251001 with the run_aita_pilot VALUES_SYS prompt "
                     "(<=3 values/comment, temp 0)",
           "seed": SEED, "n_comments": len(rows), "n_mentions": int(n_mentions),
           "smoothing": "add-one",
           "g0": {v: float(p) for v, p in zip(values, g0)}}
    G0_FILE.write_text(json.dumps(out, indent=1))
    # sanity: keys exactly match taxonomy, probs sum to 1
    back = json.loads(G0_FILE.read_text())
    assert list(back["g0"].keys()) == values and set(back["g0"]) == set(values)
    assert abs(sum(back["g0"].values()) - 1.0) < 1e-6
    print(f"wrote {G0_FILE}")

    lines = [f"| {v} | {c:.0f} | {p:.3f} |" for v, p, c in top]
    NOTE_FILE.write_text(f"""# G0: empirical base measure over the 59 moral values

**What it is.** A prior distribution G0(v) over the 59-value taxonomy
(`pipeline/values_taxonomy.yaml`, Russo et al. EACL 2026 Table 5), estimated from real
human moral rationales — mirroring Paper A, which set G0(v) = frequency of value v across
human rationales.

**How it was built.** From the HF dataset
`MattBoraske/reddit-AITA-submissions-and-comments-multiclass` (test split),
{len(rows)} top-level comments were sampled (seed {SEED}; up to {MAX_PER_POST} per post,
classification YTA/NTA only, {LEN_MIN}-{LEN_MAX} chars, bot/AutoModerator text dropped).
Each comment went to `claude-haiku-4-5-20251001` (temp 0) with the same VALUES_SYS
instruction as the AITA pilot: return at most 3 values from the fixed list that the
rationale clearly relies on. This produced {n_mentions} value mentions
({n_empty} comments yielded none). G0(v) = mention count of v / total mentions, with
**add-one smoothing** over mention counts before normalizing so all 59 values get
nonzero mass ({len(zero_vals)} values had zero raw mentions).

**Top-10 values (raw, pre-smoothing frequencies):**

| value | mentions | freq |
|---|---|---|
{chr(10).join(lines)}

**Concentration.** Top-10 mass = {t10:.3f} (raw); normalized entropy = {ent:.3f}
(smoothed; {ent_raw:.3f} on raw counts). Uniform over 59 values would be 1/59 = 0.0169
per value. Paper A's human reference: top-10 mass ~0.352 (humans) vs ~0.816 (LLMs) —
this G0 is on the human side of that gap by construction.

File: `data/g0_empirical.json` (probs sum to 1 over all 59 values). Not committed
(data/raw is gitignored; these two files are small derived artifacts).
""")
    print(f"wrote {NOTE_FILE}")
    print(f"\nn_comments={len(rows)} n_mentions={n_mentions} empty={n_empty} "
          f"zero_values={len(zero_vals)}")
    print(f"top10_mass={t10:.3f} norm_entropy_smoothed={ent:.3f} raw={ent_raw:.3f}")
    for v, p, c in top:
        print(f"  {v:24s} {c:5.0f}  {p:.3f}")
    print("zero-mention values:", zero_vals)


if __name__ == "__main__":
    {"sample": sample, "extract": extract, "aggregate": aggregate}[sys.argv[1]]()
