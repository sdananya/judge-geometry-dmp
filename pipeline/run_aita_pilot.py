"""Mini-AITA DMP pilot: rewrite -> judge (vanilla vs DMP) -> analyze.

Documented divergences from Paper A (Russo et al., EACL 2026):
- G0 is UNIFORM over the 59 extracted values (paper: empirical value frequencies from human
  rationales, which we don't have). Sensitivity to this is a known open item.
- No topic conditioning (the paper never describes its topic assignment).
- 16 FIXED profiles reused across all items (paper: resample per dilemma). Fixed profiles act
  as pseudo-annotators spanning items, which is what makes judge-instance geometry computable.
- Profile injected as top-5 (value, weight) pairs, weights renormalized (paper prompt implies
  the full 60-vector; their own examples show top-3 tuples).
- Judge = claude-haiku-4-5 (paper never names the model behind its headline DMP result).

Usage: python pipeline/run_aita_pilot.py {rewrite|judge|analyze}
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "metrics"))

OUT = ROOT / "results" / "aita_pilot"
PILOT_CSV = ROOT / "data" / "aita_pilot_150.csv"
JUDGE_MODEL = ("anthropic", "claude-haiku-4-5-20251001")
N_SAMPLES = 16
ALPHA = 10.0
SEED = 20260709

REWRITE_SYS = ("You rewrite Reddit 'Am I the Asshole' posts as abstracted third-person "
               "retellings. Remove all Reddit/AITA framing, usernames, and shorthand like "
               "'(26F)'. Refer to the author as 'the main actor' or a neutral description. "
               "Keep every morally relevant fact and the sequence of events. Keep it under "
               "250 words. Output ONLY the retelling.")
MEM_SYS = ("Answer with exactly one word, YES or NO.")
MEM_USER = ("Does the following text closely match a specific Reddit r/AmItheAsshole post "
            "you have seen during training? Consider whether you recognize this exact "
            "story.\n\n{text}")

DMP_PROMPT = """You are a moral evaluator with a personalized moral profile. Your profile consists of moral values, each associated with a numeric importance score ranging from 0 to 1. A value of 0 means the moral value has no relevance for you; a value of 1 means it is of utmost importance. Your moral judgment should be guided entirely by this profile, which reflects how much weight you place on each value. Use this profile to evaluate the following dilemma.

Your moral profile:
{profile}

Read the dilemma below and decide whether the main actor's behavior is morally ACCEPTABLE or UNACCEPTABLE. Then provide a rationale explaining your reasoning.
Output format (use exactly this):
Evaluation: <ACCEPTABLE | UNACCEPTABLE>
Rationale: <concise explanation>
-----
DILEMMA
<<<
{dilemma}
>>>"""

VANILLA_PROMPT = """You are an impartial moral evaluator. Read the dilemma below and decide whether the main actor's behavior is morally ACCEPTABLE or UNACCEPTABLE.
Then provide a rationale explaining why you made the following evaluation.

Output format (use exactly this):
Evaluation: <ACCEPTABLE | UNACCEPTABLE>
Rationale: <concise explanation>
-----
DILEMMA
<<<
{dilemma}
>>>"""


def load_values() -> list[str]:
    txt = (ROOT / "pipeline" / "values_taxonomy.yaml").read_text()
    return [m.group(1).strip() for m in re.finditer(r"^  - (.+)$", txt, re.M)]


def load_items() -> list[dict]:
    return list(csv.DictReader(open(PILOT_CSV, encoding="utf-8")))


def rewrite():
    from llm import map_calls
    items = load_items()
    jobs = [{"provider": "openai", "model": "gpt-4o-mini", "system": REWRITE_SYS,
             "user": f"TITLE: {it['title']}\n\n{it['text']}", "temperature": 0.3,
             "max_tokens": 500} for it in items]
    rewrites = map_calls(jobs, concurrency=16)
    # memorization probe x3 on the REWRITE (paper A protocol)
    mem_jobs, mem_meta = [], []
    for i, rw in enumerate(rewrites):
        for s in range(3):
            mem_jobs.append({"provider": "openai", "model": "gpt-4o-mini", "system": MEM_SYS,
                             "user": MEM_USER.format(text=rw or ""), "temperature": 1.0,
                             "max_tokens": 4, "seed": s})
            mem_meta.append(i)
    mem = map_calls(mem_jobs, concurrency=16)
    flags = {}
    for i, ans in zip(mem_meta, mem):
        flags.setdefault(i, []).append((ans or "").strip().upper().startswith("YES"))
    with open(OUT / "rewrites.jsonl", "w") as f:
        n_flagged = 0
        for i, it in enumerate(items):
            flagged = any(flags.get(i, [True]))
            n_flagged += flagged
            f.write(json.dumps({"post_id": it["post_id"], "rewrite": rewrites[i],
                                "mem_flagged": flagged}) + "\n")
    print(f"rewrote {len(items)}; memorization-flagged: {n_flagged}/{len(items)} "
          f"(paper A survival was 7.2% of raw candidates)")


def make_profiles(values: list[str]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    profiles = []
    for k in range(N_SAMPLES):
        w = rng.dirichlet(np.full(len(values), ALPHA / len(values)))
        top = np.argsort(w)[::-1][:5]
        tw = w[top] / w[top].sum()
        profiles.append({"id": k, "values": {values[i]: round(float(x), 3)
                                             for i, x in zip(top, tw)}})
    return profiles


def profile_text(p: dict) -> str:
    return "\n".join(f"- {v}: {w}" for v, w in p["values"].items())


def parse_verdict(text: str) -> tuple[int | None, str]:
    if not text:
        return None, ""
    m = re.search(r"Evaluation:\s*<?\s*(ACCEPTABLE|UNACCEPTABLE)", text, re.I)
    rat = ""
    rm = re.search(r"Rationale:\s*(.*)", text, re.S | re.I)
    if rm:
        rat = rm.group(1).strip()[:2000]
    return (None if not m else int(m.group(1).upper() == "UNACCEPTABLE")), rat


def judge():
    from llm import map_calls
    items = load_items()
    rewrites = {json.loads(l)["post_id"]: json.loads(l) for l in open(OUT / "rewrites.jsonl")}
    profiles = make_profiles(load_values())
    (OUT / "profiles.json").write_text(json.dumps(profiles, indent=1))
    provider, model = JUDGE_MODEL
    jobs, meta = [], []
    for it in items:
        dilemma = rewrites[it["post_id"]]["rewrite"]
        if not dilemma:
            continue
        for cond in ("vanilla", "dmp"):
            for s in range(N_SAMPLES):
                if cond == "vanilla":
                    user = VANILLA_PROMPT.format(dilemma=dilemma)
                else:
                    user = DMP_PROMPT.format(profile=profile_text(profiles[s]), dilemma=dilemma)
                jobs.append({"provider": provider, "model": model, "system": "",
                             "user": user, "temperature": 1.0, "max_tokens": 400, "seed": s})
                meta.append((it["post_id"], cond, s))
    print(f"{len(jobs)} judge calls")
    texts = map_calls(jobs, concurrency=48)
    n_ok = 0
    with open(OUT / "judgments.jsonl", "w") as f:
        for (pid, cond, s), text in zip(meta, texts):
            verdict, rationale = parse_verdict(text or "")
            n_ok += verdict is not None
            f.write(json.dumps({"post_id": pid, "cond": cond, "sample": s,
                                "unacceptable": verdict, "rationale": rationale,
                                "profile_id": s if cond == "dmp" else None}) + "\n")
    print(f"parsed OK: {n_ok}/{len(jobs)}")


def analyze():
    from geometry import (correlation_families, largest_principal_angle_deg,
                          noise_floor_absdiff, zscore_columns)
    items = load_items()
    rewrites = {json.loads(l)["post_id"]: json.loads(l) for l in open(OUT / "rewrites.jsonl")}
    rows = [json.loads(l) for l in open(OUT / "judgments.jsonl")]
    pids = sorted({r["post_id"] for r in rows})
    idx = {p: i for i, p in enumerate(pids)}
    it_by_pid = {it["post_id"]: it for it in items}

    mats = {c: np.full((len(pids), N_SAMPLES), np.nan) for c in ("vanilla", "dmp")}
    for r in rows:
        if r["unacceptable"] is not None:
            mats[r["cond"]][idx[r["post_id"]], r["sample"]] = r["unacceptable"]

    p_h = np.array([float(it_by_pid[p]["p_unacceptable"]) for p in pids])
    n_bin = np.array([int(it_by_pid[p]["n_binary"]) for p in pids])
    bucket = np.array([it_by_pid[p]["bucket"] for p in pids])
    ok = ~np.isnan(mats["vanilla"]).all(axis=1) & ~np.isnan(mats["dmp"]).all(axis=1)

    summary = {"n_items": int(ok.sum()), "conditions": {}}
    floor = np.array([noise_floor_absdiff(p, n, N_SAMPLES) for p, n in zip(p_h, n_bin)])
    summary["mean_noise_floor"] = round(float(floor[ok].mean()), 4)

    for cond in ("vanilla", "dmp"):
        m = mats[cond][ok]
        p_m = np.nanmean(m, axis=1)
        row = {
            "mean_abs_diff": round(float(np.mean(np.abs(p_m - p_h[ok]))), 4),
            "excess_above_floor": round(float(np.mean(np.abs(p_m - p_h[ok]) - floor[ok])), 4),
            "sigma_ratio_items": round(float(np.std(p_m) / np.std(p_h[ok])), 3),
            "mean_within_item_std": round(float(np.nanstd(m, axis=1).mean()), 3),
            "pct_unanimous_items": round(float(np.mean([len(set(x[~np.isnan(x)])) == 1 for x in m])), 3),
            "angle_ensemble_to_human_deg": round(largest_principal_angle_deg(
                zscore_columns(np.nan_to_num(m, nan=np.nanmean(m))), p_h[ok]), 1),
            "corr_p_model_p_human": round(float(np.corrcoef(p_m, p_h[ok])[0, 1]), 3),
            "mean_inter_instance_r": round(correlation_families(
                np.nan_to_num(m, nan=0.5), p_h[ok][:, None])["r_ll"], 3),
        }
        # per consensus bucket: |dP| and floor
        row["by_bucket"] = {}
        for b in sorted(set(bucket[ok])):
            sel = bucket[ok] == b
            row["by_bucket"][b] = {
                "n": int(sel.sum()),
                "abs_diff": round(float(np.mean(np.abs(p_m[sel] - p_h[ok][sel]))), 3),
                "floor": round(float(floor[ok][sel].mean()), 3),
            }
        summary["conditions"][cond] = row

    flagged = np.array([rewrites[p]["mem_flagged"] for p in pids])
    summary["mem_flagged_frac"] = round(float(flagged.mean()), 3)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


VALUES_SYS = """You classify which moral values a rationale invokes. You are given a fixed list of values and a rationale. Return a JSON array (no code fences) of the values from the list that the rationale clearly relies on, most central first, at most 3. Use EXACT spellings from the list. If none apply, return [].
VALUES: {values}"""


def extract_values():
    from llm import map_calls
    values = load_values()
    vset = set(values)
    rows = [json.loads(l) for l in open(OUT / "judgments.jsonl")]
    rows = [r for r in rows if r["rationale"]]
    jobs = [{"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
             "system": VALUES_SYS.format(values=", ".join(values)),
             "user": r["rationale"][:1500], "temperature": 0.0, "max_tokens": 100}
            for r in rows]
    print(f"{len(jobs)} extraction calls")
    texts = map_calls(jobs, concurrency=48)
    with open(OUT / "value_mentions.jsonl", "w") as f:
        for r, t in zip(rows, texts):
            try:
                vals = [v for v in json.loads((t or "[]")[(t or "[]").find("["):
                                              (t or "[]").rfind("]") + 1]) if v in vset]
            except Exception:
                vals = []
            f.write(json.dumps({"post_id": r["post_id"], "cond": r["cond"],
                                "sample": r["sample"], "values": vals}) + "\n")


def analyze_values():
    from geometry import normalized_entropy, top_k_mass
    values = load_values()
    vidx = {v: i for i, v in enumerate(values)}
    profiles = json.loads((OUT / "profiles.json").read_text())
    rows = [json.loads(l) for l in open(OUT / "value_mentions.jsonl")]
    out = {}
    for cond in ("vanilla", "dmp"):
        counts = np.zeros(len(values))
        for r in rows:
            if r["cond"] == cond:
                for v in r["values"]:
                    counts[vidx[v]] += 1
        out[cond] = {"n_mentions": int(counts.sum()),
                     "top10_mass": round(top_k_mass(counts, 10), 3),
                     "norm_entropy": round(normalized_entropy(counts), 3),
                     "n_distinct_values": int((counts > 0).sum())}
    # adherence: DMP rationale cites >=1 of its profile's top-5 values
    hits, tot = 0, 0
    for r in rows:
        if r["cond"] == "dmp" and r["values"]:
            tot += 1
            prof = set(profiles[r["sample"]]["values"].keys())
            hits += bool(prof & set(r["values"]))
    out["dmp_profile_adherence"] = round(hits / max(tot, 1), 3)
    out["adherence_chance_baseline"] = "~0.16-0.24 (2-3 random values vs 5/59 profile)"
    out["reference_paper_a"] = {"top10_mass_llm": 0.816, "top10_mass_human": 0.352,
                                "entropy_llm": 0.46, "entropy_human": 0.57,
                                "entropy_dmp": 0.52}
    (OUT / "values_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"rewrite": rewrite, "judge": judge, "analyze": analyze,
     "values": extract_values, "analyze_values": analyze_values}[sys.argv[1]]()
