"""PARIKSHA crossover (Phase 3 pre-registered "second domain" bullet, README 2026-07-14).

Question: does DMP-style conditioning (16 fixed evaluative-priority profiles) rotate judge
geometry TOWARD humans on Paper B's own public data (PARIKSHA), or only restore variance?

Design (pre-registered by construction — the 24-priority taxonomy and all analysis choices in
this file were written BEFORE any judging call was made):
  - Same 150 items as the pilot (results/pariksha_pilot/items.csv), same multi-rubric
    single-call SYSTEM/USER format and JSON schema as pipeline/run_pariksha_pilot.py.
  - vanilla16: unmodified SYSTEM, 16 samples/item.
  - conditioned16: SYSTEM prefixed with one of 16 fixed priority profiles (Dirichlet(10),
    top-5 renormalized, seed 20260714), one sample per profile per item.
  - 150 x 16 x 2 = 4,800 claude-haiku calls.

TEMPERATURE DIVERGENCE (documented, deliberate): the geometry paper / our pilot judged at
T=0.2, which is near-deterministic — a 16-sample ensemble at T=0.2 would collapse to ~1
distinct judgment and the span/variance readouts would be degenerate. Both conditions here
run at T=0.7 so that vanilla16 is a fair same-temperature baseline for conditioned16; the
comparison of interest is conditioned-vs-vanilla at matched temperature, not either vs the
paper's absolute numbers.

Pre-registered prediction (H3-crossover): conditioning recovers spread (sigma-ratio and
within-item std rise toward human levels) but the ensemble-span angle to mean-human does NOT
sit below the 2.5th percentile of a matched-variance null (vanilla per-item-rubric means +
independent gaussian noise with the conditioned condition's per-item-rubric std, 200 reps).
I.e. "pluralism as decoration": variance yes, rotation no.

Usage (repo root, .venv):
  python pipeline/pariksha_crossover.py judge     # 4,800 calls -> results/pariksha_crossover/judgments.jsonl
  python pipeline/pariksha_crossover.py analyze   # -> summary.json + printed readout
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "metrics"))

from run_pariksha_pilot import RUBRICS, SYSTEM, USER, parse_judge_json  # noqa: E402

OUT = ROOT / "results" / "pariksha_crossover"
ITEMS = ROOT / "results" / "pariksha_pilot" / "items.csv"

MODEL = ("anthropic", "claude-haiku-4-5-20251001")
N_SAMPLES = 16
TEMPERATURE = 0.7
MAX_TOKENS = 600
PROFILE_SEED = 20260714
NULL_SEED = 20260715
N_NULL = 200

# ---------------- 24 evaluative priorities (pre-registered taxonomy) ----------------
# Written before running. Each: (name, one-line gloss shown to the judge).
PRIORITIES = [
    ("Fluency-first", "prize natural, flowing language above all else"),
    ("Factual-rigor", "weight verifiable factual correctness heavily"),
    ("Cultural-context-sensitivity", "value answers grounded in the local cultural context"),
    ("Strictness", "hold answers to a demanding standard; reserve top scores"),
    ("Leniency", "give credit generously for partial success"),
    ("Completeness-focus", "value answers that cover every part of the question"),
    ("Brevity-preference", "prefer concise answers; penalize padding and repetition"),
    ("Terminology-precision", "care about exact, correct technical and domain terms"),
    ("Register-formality", "value an appropriate level of formality for the context"),
    ("Accessibility-for-laypeople", "value answers a non-expert can follow"),
    ("Source-skepticism", "distrust unsupported claims; demand grounding"),
    ("Practical-usefulness", "value actionable, usable advice over theory"),
    ("Safety-caution", "weight appropriate caveats and harm-avoidance"),
    ("Local-idiom-preference", "prefer native idiom over translated-sounding phrasing"),
    ("Grammar-pedantry", "penalize grammatical and orthographic errors sharply"),
    ("Content-over-form", "judge substance; forgive stylistic flaws"),
    ("Form-over-content", "judge presentation and style; forgive thin substance"),
    ("Benefit-of-the-doubt", "read ambiguous passages charitably"),
    ("Consistency-checking", "penalize internal contradictions within the answer"),
    ("Domain-expertise-weighting", "judge as a subject-matter expert would"),
    ("Directness", "value answers that address the question head-on, early"),
    ("Empathy-for-asker", "value tone and helpfulness toward the person asking"),
    ("Novelty-tolerance", "accept unusual framings or creative approaches"),
    ("Error-intolerance", "any clear error caps the score low"),
]
assert len(PRIORITIES) == 24

CONDITIONED_PREFIX = (
    "You are an evaluator with a personalized evaluation-priority profile. Each priority has "
    "an importance score from 0 to 1 reflecting how much weight you place on it when scoring:\n"
    "{profile}\n"
    "Let this profile guide how you weigh the rubrics' evidence.\n\n"
)


def make_profiles() -> list[list[tuple[str, float]]]:
    """16 fixed profiles: Dirichlet(alpha=10, uniform over 24), top-5 renormalized."""
    rng = np.random.default_rng(PROFILE_SEED)
    profiles = []
    for _ in range(N_SAMPLES):
        w = rng.dirichlet(np.full(len(PRIORITIES), 10.0))
        top = np.argsort(w)[::-1][:5]
        tw = w[top] / w[top].sum()
        profiles.append([(PRIORITIES[i][0], float(round(v, 3))) for i, v in zip(top, tw)])
    return profiles


def render_profile(profile: list[tuple[str, float]]) -> str:
    gloss = dict(PRIORITIES)
    return "\n".join(f"- {name} ({gloss[name]}): {w:.2f}" for name, w in profile)


def load_items() -> list[dict]:
    items = list(csv.DictReader(open(ITEMS, encoding="utf-8")))
    for it in items:
        for i in range(3):
            for r in RUBRICS:
                it[f"h{i+1}_{r}"] = int(it[f"h{i+1}_{r}"])
    return items


# ---------------- judging ----------------

def judge(items: list[dict]):
    from llm import map_calls
    profiles = make_profiles()
    (OUT / "profiles.json").write_text(json.dumps(
        [{"profile_id": s, "priorities": p} for s, p in enumerate(profiles)], indent=2))

    jobs, meta = [], []
    provider, model = MODEL
    for it in items:
        base_system = SYSTEM.format(language=it["language"])
        user = USER.format(language=it["language"], domain=it["domain"],
                           question=it["question"], answer=it["answer"])
        for s in range(N_SAMPLES):
            jobs.append({"provider": provider, "model": model, "system": base_system,
                         "user": user, "temperature": TEMPERATURE,
                         "max_tokens": MAX_TOKENS, "seed": s})
            meta.append((it["item_id"], "vanilla16", s))
        for s in range(N_SAMPLES):
            sys_c = CONDITIONED_PREFIX.format(profile=render_profile(profiles[s])) + base_system
            jobs.append({"provider": provider, "model": model, "system": sys_c,
                         "user": user, "temperature": TEMPERATURE,
                         "max_tokens": MAX_TOKENS, "seed": s})
            meta.append((it["item_id"], "conditioned16", s))

    print(f"{len(jobs)} judge calls ({len(items)} items x {N_SAMPLES} x 2 conds)")
    texts = map_calls(jobs, concurrency=48)
    n_ok = 0
    with open(OUT / "judgments.jsonl", "w") as f:
        for (item_id, cond, s), text in zip(meta, texts):
            parsed = parse_judge_json(text)
            n_ok += parsed is not None
            f.write(json.dumps({
                "item_id": item_id, "cond": cond, "sample": s,
                "scores": {k: parsed[k] for k in RUBRICS} if parsed else None,
                "raw": (text or "")[:400]}, ensure_ascii=False) + "\n")
    print(f"parsed OK: {n_ok}/{len(jobs)}")


# ---------------- analysis ----------------

def stack_z(m: np.ndarray) -> np.ndarray:
    """Per-rubric z-score across items, then flatten (identical recipe to the pilot)."""
    from geometry import zscore_columns
    return zscore_columns(m).reshape(-1)


def analyze(items: list[dict]):
    from geometry import largest_principal_angle_deg, sigma_ratio

    idx = {it["item_id"]: i for i, it in enumerate(items)}
    n = len(items)
    judgments = [json.loads(l) for l in open(OUT / "judgments.jsonl", encoding="utf-8")]

    # tensors: cond -> samples x items x rubrics
    T = {c: np.full((N_SAMPLES, n, 3), np.nan) for c in ("vanilla16", "conditioned16")}
    n_calls = {c: 0 for c in T}
    n_parsed = {c: 0 for c in T}
    for j in judgments:
        n_calls[j["cond"]] += 1
        if j["scores"]:
            n_parsed[j["cond"]] += 1
            T[j["cond"]][j["sample"], idx[j["item_id"]]] = [j["scores"][r] for r in RUBRICS]

    H = np.stack([[[it[f"h{i+1}_{r}"] for r in RUBRICS] for it in items]
                  for i in range(3)])          # 3 x n x 3
    h_mean = H.mean(axis=0)                    # n x 3
    h_pool = H.reshape(-1, 3)                  # pooled individual raters

    summary = {"n_items": n, "temperature": TEMPERATURE, "conditions": {}}

    # complete-case item sets (all 16 samples parsed) — angle computations use the
    # INTERSECTION so both conditions and the null share one item set.
    complete = {c: ~np.isnan(T[c]).any(axis=(0, 2)) for c in T}
    both_ok = complete["vanilla16"] & complete["conditioned16"]
    summary["n_items_complete_both"] = int(both_ok.sum())

    for c, t in T.items():
        row = {"parse_rate": round(n_parsed[c] / max(n_calls[c], 1), 4),
               "n_calls": n_calls[c], "n_items_complete": int(complete[c].sum())}
        flat = t.reshape(-1, 3)  # all samples pooled
        for k, r in enumerate(RUBRICS):
            col = flat[:, k]
            row[f"sigma_ratio_pool_{r}"] = round(sigma_ratio(col[~np.isnan(col)], h_pool[:, k]), 3)
        # spread readout: mean within-item-per-rubric std across the 16 samples
        wstd = {}
        for k, r in enumerate(RUBRICS):
            per_item = []
            for i in range(n):
                v = t[:, i, k]
                v = v[~np.isnan(v)]
                if v.size >= 2:
                    per_item.append(v.std())
            wstd[r] = round(float(np.mean(per_item)), 3)
        row["within_item_std"] = wstd
        row["within_item_std_mean"] = round(float(np.mean(list(wstd.values()))), 3)

        # ensemble geometry on the shared complete set
        cols = np.column_stack([stack_z(t[s][both_ok]) for s in range(N_SAMPLES)])
        hv = stack_z(h_mean[both_ok])
        row["ensemble_span_angle_deg"] = round(largest_principal_angle_deg(cols, hv), 1)
        mean_vec = stack_z(t[:, both_ok].mean(axis=0))
        row["ensemble_mean_angle_deg"] = round(largest_principal_angle_deg(mean_vec, hv), 1)
        # inter-instance correlation: are the 16 instances distinct judges or one?
        zs = [stack_z(t[s][both_ok]) for s in range(N_SAMPLES)]
        rs = [float(np.corrcoef(zs[a], zs[b])[0, 1])
              for a in range(N_SAMPLES) for b in range(a + 1, N_SAMPLES)]
        row["inter_instance_r_mean"] = round(float(np.mean(rs)), 3)
        row["inter_instance_r_range"] = [round(min(rs), 3), round(max(rs), 3)]
        summary["conditions"][c] = row

    # human floors on the same item set
    Hs = H[:, both_ok]
    summary["human_loo_angle_deg"] = [round(largest_principal_angle_deg(
        stack_z(Hs[a]), stack_z(np.mean([Hs[b] for b in range(3) if b != a], axis=0))), 1)
        for a in range(3)]
    summary["human_pair_angle_deg"] = [round(largest_principal_angle_deg(
        stack_z(Hs[a]), stack_z(Hs[b])), 1) for a in range(3) for b in range(a + 1, 3)]
    hz = [stack_z(Hs[a]) for a in range(3)]
    summary["human_pair_r"] = [round(float(np.corrcoef(hz[a], hz[b])[0, 1]), 3)
                               for a in range(3) for b in range(a + 1, 3)]

    # ---------------- matched-variance null for the conditioned span angle ----------------
    # 200 synthetic ensembles: 16 columns of (vanilla per-item-rubric MEAN + independent
    # gaussian noise with the CONDITIONED condition's per-item-rubric STD). Same angle
    # computation. If conditioning only adds item-local variance around the vanilla judgment,
    # the observed conditioned span angle should be an ordinary draw from this null.
    v_mean = np.nanmean(T["vanilla16"][:, both_ok], axis=0)          # n_ok x 3
    c_std = np.nanstd(T["conditioned16"][:, both_ok], axis=0)        # n_ok x 3
    hv = stack_z(h_mean[both_ok])
    rng = np.random.default_rng(NULL_SEED)
    null_angles = []
    for _ in range(N_NULL):
        noise = rng.normal(0.0, 1.0, size=(N_SAMPLES,) + v_mean.shape) * c_std[None]
        cols = np.column_stack([stack_z(v_mean + noise[s]) for s in range(N_SAMPLES)])
        null_angles.append(largest_principal_angle_deg(cols, hv))
    null_angles = np.array(null_angles)
    obs = summary["conditions"]["conditioned16"]["ensemble_span_angle_deg"]
    summary["null"] = {
        "n": N_NULL,
        "mean": round(float(null_angles.mean()), 1),
        "p2.5": round(float(np.percentile(null_angles, 2.5)), 1),
        "p97.5": round(float(np.percentile(null_angles, 97.5)), 1),
        "observed_conditioned_angle": obs,
        "observed_percentile": round(float((null_angles < obs).mean() * 100), 1),
        "beats_null": bool(obs < np.percentile(null_angles, 2.5)),
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    cmd = sys.argv[1]
    if cmd == "judge":
        judge(items)
    elif cmd == "analyze":
        analyze(items)
    else:
        raise SystemExit("usage: pariksha_crossover.py judge|analyze")


if __name__ == "__main__":
    main()
