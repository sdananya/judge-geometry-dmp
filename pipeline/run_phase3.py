"""Phase 3 (paper must-haves; see README pre-registration 2026-07-14).

New vs Phase 2: n=500 items; faithful-DMP (per-item resampled, N=32, topic-conditioned);
distribution-elicitation + few-shot elicitation baselines; profile-set reseeds; expanded
judge roster; T=1.5 control on OpenAI; TOST equivalence analysis; council aggregate.

Subcommands:
  judge_core      haiku full grid on 500 items (vanilla/dmp_empirical/shuffled_g0/
                  random_persona x16, dmp_faithful x32, elicit/fewshot_elicit x3)
  judge_reseeds   dmp_empirical with profile-set seeds B/C on the 150-item subset
  judge_alt SPEC  SPEC=provider:model:tag  -> vanilla/dmp_empirical/random_persona x8
                  + elicit x3 on the 150-item subset
  judge_temp      gpt-4o-mini vanilla @ T=1.5 x16 on the 150 subset
  analyze         everything: per-condition metrics, TOST, decomposition, council
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

from run_phase2 import (DMP_PROMPT, PERSONA_PROMPT, PERSONA_POOL, VANILLA_PROMPT,
                        load_values, parse_verdict, profile_text)

OUT = ROOT / "results" / "phase3"
P2 = ROOT / "results" / "phase2"
SEED = 20260714
ALPHA = 10.0
N_FIXED = 16
N_FAITHFUL = 32
TOST_DELTA = 0.05
HAIKU = ("anthropic", "claude-haiku-4-5-20251001")

ELICIT_PROMPT = """You are estimating public opinion. Read the dilemma below and estimate what percentage of ordinary people would judge the main actor's behavior morally UNACCEPTABLE.
Output format (use exactly this):
Percentage: <integer between 0 and 100>
-----
DILEMMA
<<<
{dilemma}
>>>"""

FEWSHOT_HEADER = """You are estimating public opinion. For each dilemma, estimate what percentage of ordinary people would judge the main actor's behavior morally UNACCEPTABLE. Here are examples with the true percentages measured from large groups of real people:

{examples}

Now estimate for the new dilemma below.
Output format (use exactly this):
Percentage: <integer between 0 and 100>
-----
DILEMMA
<<<
{dilemma}
>>>"""


# ---------------- data ----------------

def load_items_500():
    return list(csv.DictReader(open(ROOT / "data" / "aita_pilot_500.csv", encoding="utf-8")))


def load_rewrites():
    return {json.loads(l)["post_id"]: json.loads(l) for l in open(OUT / "rewrites.jsonl")}


def subset_150(items):
    old = {r["post_id"] for r in csv.DictReader(open(ROOT / "data" / "aita_pilot_150.csv"))}
    return [it for it in items if it["post_id"] in old]


def fixed_profiles(g0_vec, names, seed):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(N_FIXED):
        w = rng.dirichlet(ALPHA * g0_vec)
        top = np.argsort(w)[::-1][:5]
        tw = w[top] / w[top].sum()
        out.append({"id": k, "values": {names[i]: round(float(x), 3) for i, x in zip(top, tw)}})
    return out


def build_fixed_profiles():
    values = load_values()
    n = len(values)
    g0 = json.loads((ROOT / "data" / "g0_empirical.json").read_text())["g0"]
    g0v = np.array([g0[v] for v in values])
    rng = np.random.default_rng(SEED)
    g0_shuf = g0v[rng.permutation(n)]
    return {
        "dmp_empirical": fixed_profiles(g0v, values, SEED + 1),
        "dmp_empirical_seedB": fixed_profiles(g0v, values, SEED + 101),
        "dmp_empirical_seedC": fixed_profiles(g0v, values, SEED + 202),
        "shuffled_g0": fixed_profiles(g0_shuf, values, SEED + 3),
        "random_persona": fixed_profiles(np.full(n, 1.0 / n), PERSONA_POOL[:n], SEED + 4),
    }


def faithful_profile(topic, item_idx, s, g0_topics, values):
    g = g0_topics["topics"].get(topic) or g0_topics["global"]
    gv = np.array([g[v] for v in values])
    rng = np.random.default_rng([SEED, 777, item_idx, s])
    w = rng.dirichlet(ALPHA * gv)
    top = np.argsort(w)[::-1][:5]
    tw = w[top] / w[top].sum()
    return {"values": {values[i]: round(float(x), 3) for i, x in zip(top, tw)}}


def parse_percent(text):
    m = re.search(r"Percentage:\s*(\d{1,3})", text or "")
    if not m:
        m = re.search(r"\b(\d{1,3})\s*%", text or "")
    if not m:
        return None
    v = int(m.group(1))
    return v / 100 if 0 <= v <= 100 else None


# ---------------- judging ----------------

def _write(rows_out, fname):
    with open(OUT / fname, "a") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")


def judge_core():
    from llm import map_calls
    gate = json.loads((OUT / "rewrite_gate.json").read_text())
    assert gate["gate_ok"], f"rewrite gate FAILED: {gate}"
    items = load_items_500()
    rw = load_rewrites()
    topics = json.loads((ROOT / "data" / "topics_500.json").read_text())
    g0_topics = json.loads((ROOT / "data" / "g0_topics.json").read_text())
    values = load_values()
    profs = build_fixed_profiles()
    (OUT / "profiles.json").write_text(json.dumps(profs, indent=1))
    fewshot = json.loads((ROOT / "data" / "fewshot_pool.json").read_text())
    ex = "\n\n".join(f"DILEMMA:\n{e['rewrite'][:900]}\nPercentage: {e['pct_unacceptable']}"
                     for e in fewshot)
    provider, model = HAIKU
    jobs, meta = [], []
    for idx, it in enumerate(items):
        d = rw[it["post_id"]]["rewrite"]
        if not d or not rw[it["post_id"]]["anchored"]:
            continue
        for cond in ("vanilla", "dmp_empirical", "shuffled_g0", "random_persona"):
            for s in range(N_FIXED):
                if cond == "vanilla":
                    user = VANILLA_PROMPT.format(dilemma=d)
                elif cond == "random_persona":
                    user = PERSONA_PROMPT.format(profile=profile_text(profs[cond][s]), dilemma=d)
                else:
                    user = DMP_PROMPT.format(profile=profile_text(profs[cond][s]), dilemma=d)
                jobs.append(dict(provider=provider, model=model, system="", user=user,
                                 temperature=1.0, max_tokens=400, seed=s))
                meta.append((it["post_id"], cond, s, "verdict"))
        for s in range(N_FAITHFUL):
            p = faithful_profile(topics[it["post_id"]], idx, s, g0_topics, values)
            jobs.append(dict(provider=provider, model=model, system="",
                             user=DMP_PROMPT.format(profile=profile_text(p), dilemma=d),
                             temperature=1.0, max_tokens=400, seed=s))
            meta.append((it["post_id"], "dmp_faithful", s, "verdict"))
        for cond, tmpl in (("elicit", ELICIT_PROMPT),
                           ("fewshot_elicit", None)):
            for s in range(3):
                user = (ELICIT_PROMPT.format(dilemma=d) if cond == "elicit"
                        else FEWSHOT_HEADER.format(examples=ex, dilemma=d))
                jobs.append(dict(provider=provider, model=model, system="", user=user,
                                 temperature=1.0, max_tokens=30, seed=s))
                meta.append((it["post_id"], cond, s, "percent"))
    print(f"{len(jobs)} calls")
    texts = map_calls(jobs, concurrency=48)
    rows_out, n_ok = [], 0
    for (pid, cond, s, kind), text in zip(meta, texts):
        if kind == "verdict":
            v, rat = parse_verdict(text or "")
            n_ok += v is not None
            rows_out.append({"post_id": pid, "cond": cond, "sample": s,
                             "unacceptable": v, "rationale": rat[:800]})
        else:
            p = parse_percent(text)
            n_ok += p is not None
            rows_out.append({"post_id": pid, "cond": cond, "sample": s, "p_unacc": p})
    (OUT / "judgments_core.jsonl").write_text("")
    _write(rows_out, "judgments_core.jsonl")
    print(f"parsed OK: {n_ok}/{len(jobs)}")


def judge_reseeds():
    from llm import map_calls
    items = subset_150(load_items_500())
    rw = load_rewrites()
    profs = build_fixed_profiles()
    provider, model = HAIKU
    jobs, meta = [], []
    for it in items:
        d = rw[it["post_id"]]["rewrite"]
        for cond in ("dmp_empirical_seedB", "dmp_empirical_seedC"):
            for s in range(N_FIXED):
                jobs.append(dict(provider=provider, model=model, system="",
                                 user=DMP_PROMPT.format(profile=profile_text(profs[cond][s]),
                                                        dilemma=d),
                                 temperature=1.0, max_tokens=400, seed=s))
                meta.append((it["post_id"], cond, s))
    print(f"{len(jobs)} calls")
    texts = map_calls(jobs, concurrency=48)
    rows_out = []
    for (pid, cond, s), text in zip(meta, texts):
        v, rat = parse_verdict(text or "")
        rows_out.append({"post_id": pid, "cond": cond, "sample": s,
                         "unacceptable": v, "rationale": ""})
    (OUT / "judgments_reseeds.jsonl").write_text("")
    _write(rows_out, "judgments_reseeds.jsonl")


def judge_alt(spec):
    from llm import map_calls
    provider, model, tag = spec.split(":")
    items = subset_150(load_items_500())
    rw = load_rewrites()
    profs = build_fixed_profiles()
    jobs, meta = [], []
    for it in items:
        d = rw[it["post_id"]]["rewrite"]
        for cond in ("vanilla", "dmp_empirical", "random_persona"):
            for s in range(8):
                if cond == "vanilla":
                    user = VANILLA_PROMPT.format(dilemma=d)
                elif cond == "random_persona":
                    user = PERSONA_PROMPT.format(profile=profile_text(profs[cond][s]), dilemma=d)
                else:
                    user = DMP_PROMPT.format(profile=profile_text(profs[cond][s]), dilemma=d)
                jobs.append(dict(provider=provider, model=model, system="", user=user,
                                 temperature=1.0, max_tokens=400, seed=s))
                meta.append((it["post_id"], cond, s, "verdict"))
        for s in range(3):
            jobs.append(dict(provider=provider, model=model, system="",
                             user=ELICIT_PROMPT.format(dilemma=d),
                             temperature=1.0, max_tokens=30, seed=s))
            meta.append((it["post_id"], "elicit", s, "percent"))
    print(f"{len(jobs)} calls -> judgments_{tag}.jsonl")
    texts = map_calls(jobs, concurrency=16)
    rows_out = []
    for (pid, cond, s, kind), text in zip(meta, texts):
        if kind == "verdict":
            v, _ = parse_verdict(text or "")
            rows_out.append({"post_id": pid, "cond": cond, "sample": s, "unacceptable": v})
        else:
            rows_out.append({"post_id": pid, "cond": cond, "sample": s,
                             "p_unacc": parse_percent(text)})
    (OUT / f"judgments_{tag}.jsonl").write_text("")
    _write(rows_out, f"judgments_{tag}.jsonl")


def judge_temp():
    from llm import map_calls
    items = subset_150(load_items_500())
    rw = load_rewrites()
    jobs, meta = [], []
    for it in items:
        d = rw[it["post_id"]]["rewrite"]
        for s in range(N_FIXED):
            jobs.append(dict(provider="openai", model="gpt-4o-mini", system="",
                             user=VANILLA_PROMPT.format(dilemma=d),
                             temperature=1.5, max_tokens=400, seed=s))
            meta.append((it["post_id"], "vanilla_T1.5", s))
    texts = map_calls(jobs, concurrency=16)
    rows_out = []
    for (pid, cond, s), text in zip(meta, texts):
        v, _ = parse_verdict(text or "")
        rows_out.append({"post_id": pid, "cond": cond, "sample": s, "unacceptable": v})
    (OUT / "judgments_temp.jsonl").write_text("")
    _write(rows_out, "judgments_temp.jsonl")


# ---------------- analysis ----------------

def analyze():
    from geometry import noise_floor_absdiff, stretch_rotate_residual
    items = load_items_500()
    it_by = {it["post_id"]: it for it in items}
    rows = [json.loads(l) for l in open(OUT / "judgments_core.jsonl")]
    pids = sorted({r["post_id"] for r in rows})
    idx = {p: i for i, p in enumerate(pids)}
    p_h = np.array([float(it_by[p]["p_unacceptable"]) for p in pids])
    n_bin = np.array([int(it_by[p]["n_binary"]) for p in pids])

    def pm_for(cond, source_rows):
        acc = {}
        for r in source_rows:
            if r["cond"] != cond:
                continue
            if "p_unacc" in r:
                if r["p_unacc"] is not None:
                    acc.setdefault(r["post_id"], []).append(r["p_unacc"])
            elif r["unacceptable"] is not None:
                acc.setdefault(r["post_id"], []).append(r["unacceptable"])
        return acc

    conds = ["vanilla", "dmp_empirical", "shuffled_g0", "random_persona",
             "dmp_faithful", "elicit", "fewshot_elicit"]
    pm = {}
    for c in conds:
        acc = pm_for(c, rows)
        pm[c] = np.array([np.mean(acc.get(p, [np.nan])) for p in pids])

    rng = np.random.default_rng(SEED)
    B = 1000
    bi = rng.integers(0, len(pids), size=(B, len(pids)))

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    summary = {"n_items": len(pids), "conditions": {}, "tost": {}, "decomposition": {}}
    floor = np.array([noise_floor_absdiff(p, n, 16) for p, n in zip(p_h, n_bin)])
    summary["mean_noise_floor_n16"] = round(float(floor.mean()), 4)
    for c in conds:
        ok = np.isfinite(pm[c])
        summary["conditions"][c] = {
            "n": int(ok.sum()),
            "corr_with_human": round(corr(pm[c], p_h), 3),
            "mean_abs_diff": round(float(np.nanmean(np.abs(pm[c] - p_h))), 4),
        }
    # TOST vs vanilla (delta pre-registered 0.05)
    for c in [x for x in conds if x != "vanilla"]:
        dc = []
        for b in bi:
            dc.append(corr(pm[c][b], p_h[b]) - corr(pm["vanilla"][b], p_h[b]))
        lo, hi = np.percentile(dc, [2.5, 97.5])
        summary["tost"][f"{c}_minus_vanilla"] = {
            "delta": round(float(np.mean(dc)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "equivalent_within_0.05": bool(lo > -TOST_DELTA and hi < TOST_DELTA),
            "significant": bool(lo > 0 or hi < 0),
        }
    # decomposition vs vanilla
    for c in [x for x in conds if x != "vanilla"]:
        ok = np.isfinite(pm[c]) & np.isfinite(pm["vanilla"])
        d = stretch_rotate_residual(pm[c][ok] - pm["vanilla"][ok], pm["vanilla"][ok], p_h[ok])
        summary["decomposition"][c] = {k: round(v, 4) for k, v in d.items()
                                       if k != "total_energy"}
    # reseeds
    if (OUT / "judgments_reseeds.jsonl").exists():
        rrows = [json.loads(l) for l in open(OUT / "judgments_reseeds.jsonl")]
        sub = sorted({r["post_id"] for r in rrows})
        ph_s = np.array([float(it_by[p]["p_unacceptable"]) for p in sub])
        summary["reseeds"] = {}
        for c in ("dmp_empirical_seedB", "dmp_empirical_seedC"):
            acc = pm_for(c, rrows)
            v = np.array([np.mean(acc.get(p, [np.nan])) for p in sub])
            summary["reseeds"][c] = round(corr(v, ph_s), 3)
    # alt judges + temp + council
    alt = {}
    for f in sorted(OUT.glob("judgments_*.jsonl")):
        tag = f.stem.replace("judgments_", "")
        if tag in ("core", "reseeds", "temp"):
            continue
        arows = [json.loads(l) for l in open(f)]
        sub = sorted({r["post_id"] for r in arows})
        ph_s = np.array([float(it_by[p]["p_unacceptable"]) for p in sub])
        alt[tag] = {}
        for c in ("vanilla", "dmp_empirical", "random_persona", "elicit"):
            acc = pm_for(c, arows)
            v = np.array([np.mean(acc.get(p, [np.nan])) for p in sub])
            alt[tag][c] = round(corr(v, ph_s), 3)
    summary["alt_judges"] = alt
    if (OUT / "judgments_temp.jsonl").exists():
        trows = [json.loads(l) for l in open(OUT / "judgments_temp.jsonl")]
        sub = sorted({r["post_id"] for r in trows})
        ph_s = np.array([float(it_by[p]["p_unacceptable"]) for p in sub])
        acc = pm_for("vanilla_T1.5", trows)
        v = np.array([np.mean(acc.get(p, [np.nan])) for p in sub])
        wstd = np.array([np.std(acc[p]) for p in sub if p in acc])
        summary["temp_control_gpt4omini_T1.5"] = {
            "corr": round(corr(v, ph_s), 3), "within_item_std": round(float(wstd.mean()), 3)}
    # council: pool vanilla verdict samples across all judges on the 150 subset
    pool = {}
    for f in sorted(OUT.glob("judgments_*.jsonl")) + [OUT / "judgments_core.jsonl"]:
        for l in open(f):
            r = json.loads(l)
            if r["cond"] == "vanilla" and r.get("unacceptable") is not None:
                pool.setdefault(r["post_id"], []).append(r["unacceptable"])
    sub = sorted(subset_150(items)[i]["post_id"] for i in range(len(subset_150(items))))
    v = np.array([np.mean(pool.get(p, [np.nan])) for p in sub])
    ph_s = np.array([float(it_by[p]["p_unacceptable"]) for p in sub])
    summary["council_vanilla_all_judges"] = {"corr": round(corr(v, ph_s), 3),
                                             "judges_pooled": "all files with vanilla"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "judge_alt":
        judge_alt(sys.argv[2])
    else:
        {"judge_core": judge_core, "judge_reseeds": judge_reseeds,
         "judge_temp": judge_temp, "analyze": analyze}[cmd]()
