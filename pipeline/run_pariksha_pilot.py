"""PARIKSHA pilot: select 150 items -> judge with 7 LLMs -> geometric diagnostics.

Usage (from repo root, .venv):
  python pipeline/run_pariksha_pilot.py build     # select items -> results/pariksha_pilot/items.csv
  python pipeline/run_pariksha_pilot.py judge     # call judges -> judgments.jsonl (+ transcripts.html)
  python pipeline/run_pariksha_pilot.py analyze   # -> summary.json / summary.md

Reference values from the geometry paper (arXiv:2606.03043, DLQ panels) we compare against:
  sigma_J/sigma_H subjective ~0.73-0.78; judge-vs-human angle subjective ~71-74 deg,
  human-pair floor 77.8-80.8; objective rubric: judge 58.5 vs human floor 61.7;
  r_ll ~0.35 > r_lh ~0.27-0.32, r_hh ~0.36.
"""

from __future__ import annotations

import csv
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "metrics"))

RAW = ROOT / "data" / "raw" / "pariksha-microsoft"
OUT = ROOT / "results" / "pariksha_pilot"

# gujarati/punjabi TSVs are label-swapped upstream; irrelevant here (we use hi/te/ml) but
# kept explicit so nobody reuses this loader without the fix.
TSV_LANG_FIX = {"gujarati": "punjabi", "punjabi": "gujarati"}

LANGS = {"hindi": 60, "telugu": 45, "malayalam": 45}
DOMAIN_SPLIT = {"culture": 0.5, "finance": 0.25, "health": 0.25}
MODEL_CAP = 4
SEED = 20260709
RUBRICS = ["ling", "task", "halluc"]

JUDGES = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-sonnet-4-5"),
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("openrouter", "google/gemma-3-27b-it"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    ("openrouter", "qwen/qwen3-235b-a22b"),
]

SYSTEM = """You are an expert evaluator of AI assistant answers in Indian languages.
Rate the ANSWER to the QUESTION on three rubrics:
1. linguistic_acceptability: is the answer written in natural, grammatical, appropriate {language}? 2 = fully acceptable, 1 = somewhat acceptable, 0 = not acceptable.
2. task_quality: does the answer correctly and completely address the question? 2 = good, 1 = somewhat, 0 = poor.
3. hallucination: does the answer contain fabricated, false, or unsupported content? 1 = yes (hallucination present), 0 = no.
Do not browse. Use only general knowledge and the given text. Evaluate the answer in its own language; keep explanations in concise English.
Output EXACTLY one minified JSON object, no code fences, in this schema:
{{"linguistic_acceptability": {{"rating": <0|1|2>, "explanation": "<short>"}}, "task_quality": {{"rating": <0|1|2>, "explanation": "<short>"}}, "hallucination": {{"rating": <0|1>, "explanation": "<short>"}}}}"""

USER = """Language: {language}
Domain: {domain}

QUESTION:
{question}

ANSWER:
{answer}"""


# ---------------- loading ----------------

def load_human_rows(lang: str) -> list[dict]:
    rows, seen = [], set()
    for suffix in ("_just", ""):
        f = RAW / "karya_eval" / "round1" / "individual" / f"{lang}_ratings_output_indi{suffix}.tsv"
        if not f.exists():
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                key = (r["model"], r["prompt_id"], r["worker_id"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "model": r["model"], "prompt_id": r["prompt_id"], "worker_id": r["worker_id"],
                    "ling": int(r["ling_rating"]), "task": int(r["task_rating"]),
                    "halluc": 1 if str(r["hallucination_rating"]).strip().lower() == "true" else 0,
                })
    return rows


def load_answers(lang: str) -> dict:
    data = json.loads((RAW / "outputs" / "round1" / "formatted" / f"output_{lang}.json").read_text())
    items = {}
    meta_keys = {"prompt_id", "prompt_timestamp", "prompt_creator", "prompt_type", "language",
                 "system_prompt", "prompt"}
    for obj in data:
        for k, v in obj.items():
            if k in meta_keys or not isinstance(v, dict) or "response" not in v:
                continue
            items[(obj["prompt_id"], k)] = {
                "domain": obj.get("prompt_type", "?"), "question": obj["prompt"],
                "answer": v["response"] or "",
            }
    return items


def build_items() -> list[dict]:
    rng = random.Random(SEED)
    selected = []
    for lang, n_target in LANGS.items():
        human = load_human_rows(lang)
        answers = load_answers(lang)
        by_item = defaultdict(list)
        for r in human:
            by_item[(r["prompt_id"], r["model"])].append(r)
        candidates = []
        for key, raters in by_item.items():
            if len(raters) >= 3 and key in answers and answers[key]["answer"].strip():
                candidates.append((key, raters[:3], answers[key]))
        rng.shuffle(candidates)
        want = {d: round(frac * n_target) for d, frac in DOMAIN_SPLIT.items()}
        got = defaultdict(int)
        per_model = defaultdict(int)
        chosen = []

        def take(c, relax_domain=False):
            (pid, model), raters, ans = c
            d = ans["domain"]
            if not relax_domain and got.get(d, 0) >= want.get(d, 0):
                return False
            if per_model[(lang, model)] >= MODEL_CAP:
                return False
            got[d] += 1
            per_model[(lang, model)] += 1
            chosen.append(c)
            return True

        for c in candidates:
            if len(chosen) >= n_target:
                break
            take(c)
        for c in candidates:  # top-up ignoring domain targets if strata ran dry
            if len(chosen) >= n_target:
                break
            if c not in chosen:
                take(c, relax_domain=True)

        for (pid, model), raters, ans in chosen:
            selected.append({
                "item_id": f"{lang}|{pid}|{model}", "language": lang, "prompt_id": pid,
                "gen_model": model, "domain": ans["domain"],
                "question": ans["question"], "answer": ans["answer"],
                **{f"h{i+1}_{rub}": raters[i][rub] for i in range(3) for rub in RUBRICS},
            })
        print(f"{lang}: {len(chosen)}/{n_target} items, domains={dict(got)}")
    return selected


# ---------------- judging ----------------

def parse_judge_json(text: str) -> dict | None:
    if not text:
        return None
    s = text[text.find("{"): text.rfind("}") + 1]
    try:
        obj = json.loads(s)
        return {
            "ling": int(obj["linguistic_acceptability"]["rating"]),
            "task": int(obj["task_quality"]["rating"]),
            "halluc": int(obj["hallucination"]["rating"]),
            "expl": {k: obj[kk].get("explanation", "") for k, kk in
                     [("ling", "linguistic_acceptability"), ("task", "task_quality"),
                      ("halluc", "hallucination")]},
        }
    except Exception:
        return None


def judge(items: list[dict]):
    from llm import map_calls
    jobs, meta = [], []
    for it in items:
        for provider, model in JUDGES:
            jobs.append({
                "provider": provider, "model": model,
                "system": SYSTEM.format(language=it["language"]),
                "user": USER.format(language=it["language"], domain=it["domain"],
                                    question=it["question"], answer=it["answer"]),
                "temperature": 0.2, "max_tokens": 600,
            })
            meta.append((it["item_id"], f"{provider}/{model}"))
    print(f"{len(jobs)} judge calls ({len(items)} items x {len(JUDGES)} judges)")
    texts = map_calls(jobs, concurrency=24)
    n_ok = 0
    with open(OUT / "judgments.jsonl", "w") as f:
        for (item_id, judge_id), text in zip(meta, texts):
            parsed = parse_judge_json(text)
            n_ok += parsed is not None
            f.write(json.dumps({"item_id": item_id, "judge": judge_id,
                                "scores": {k: parsed[k] for k in RUBRICS} if parsed else None,
                                "expl": parsed["expl"] if parsed else None,
                                "raw": text}, ensure_ascii=False) + "\n")
    print(f"parsed OK: {n_ok}/{len(jobs)}")


def write_html(items: list[dict], judgments: list[dict], n: int = 20):
    by_item = defaultdict(list)
    for j in judgments:
        by_item[j["item_id"]].append(j)
    parts = ["<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:900px;margin:auto'>"]
    for it in items[:n]:
        parts.append(f"<h3>{html.escape(it['item_id'])} [{it['domain']}]</h3>")
        parts.append(f"<p><b>Q:</b> {html.escape(it['question'])}</p>")
        parts.append(f"<p style='background:#f6f6f6;padding:8px'><b>A:</b> {html.escape(it['answer'][:2500])}</p>")
        hrow = " / ".join(f"h{i+1}: L{it[f'h{i+1}_ling']} T{it[f'h{i+1}_task']} H{it[f'h{i+1}_halluc']}" for i in range(3))
        parts.append(f"<p><b>humans:</b> {hrow}</p><ul>")
        for j in sorted(by_item[it["item_id"]], key=lambda x: x["judge"]):
            s = j["scores"]
            lab = f"L{s['ling']} T{s['task']} H{s['halluc']}" if s else "PARSE-FAIL"
            ex = html.escape(json.dumps(j.get("expl") or {}, ensure_ascii=False)[:300])
            parts.append(f"<li><b>{html.escape(j['judge'])}</b>: {lab} <small>{ex}</small></li>")
        parts.append("</ul><hr>")
    parts.append("</body></html>")
    (OUT / "transcripts.html").write_text("".join(parts), encoding="utf-8")


# ---------------- analysis ----------------

def analyze(items: list[dict], judgments: list[dict]):
    from geometry import (correlation_families, effective_rank_r95,
                          largest_principal_angle_deg, sigma_ratio, zscore_columns)

    idx = {it["item_id"]: i for i, it in enumerate(items)}
    n = len(items)
    judge_ids = sorted({j["judge"] for j in judgments if j["scores"]})
    # score tensors: judges/humans x items x rubrics
    J = {jid: np.full((n, 3), np.nan) for jid in judge_ids}
    for j in judgments:
        if j["scores"]:
            J[j["judge"]][idx[j["item_id"]]] = [j["scores"][r] for r in RUBRICS]
    H = np.stack([[[it[f"h{i+1}_{r}"] for r in RUBRICS] for it in items] for i in range(3)])  # 3 x n x 3
    h_mean = H.mean(axis=0)                          # n x 3
    h_pool = H.reshape(-1, 3)                        # individual-rater pool

    summary = {"n_items": n, "judges": {}, "humans": {}, "reference": {
        "paper_sigma_subj": "0.73-0.78", "paper_angle_subj": "71.6-74.0",
        "paper_human_floor_subj": "77.8-80.8", "paper_obj": "judge 58.5 vs human 61.7",
        "paper_r": "r_ll~0.35 r_lh~0.27-0.32 r_hh~0.36"}}

    def stack_z(m):
        """Z-score each rubric column, then flatten — angles must not be driven by the
        rubrics' different scales/means (0-2 vs 0-1), only by across-item variation."""
        return zscore_columns(m).reshape(-1)

    # human-pair floors (angle between rater stacked vectors; per-rubric z-scored)
    floors = []
    for a in range(3):
        for b in range(a + 1, 3):
            floors.append(largest_principal_angle_deg(stack_z(H[a]), stack_z(H[b])))
    summary["humans"]["pair_angle_deg"] = [round(f, 1) for f in floors]
    # leave-one-out floor: rater vs mean of the other two — the fair benchmark for a
    # single judge scored against mean-human
    loo = [largest_principal_angle_deg(
        stack_z(H[a]), stack_z(np.mean([H[b] for b in range(3) if b != a], axis=0)))
        for a in range(3)]
    summary["humans"]["loo_angle_deg"] = [round(f, 1) for f in loo]
    # per-rubric human-pair angles (subjective-vs-objective contrast lives here)
    summary["humans"]["pair_angle_per_rubric"] = {
        r: round(float(np.mean([largest_principal_angle_deg(H[a][:, k], H[b][:, k])
                                for a in range(3) for b in range(a + 1, 3)])), 1)
        for k, r in enumerate(RUBRICS)}
    per_rubric_hh = {}
    for k, r in enumerate(RUBRICS):
        vals = [float(np.corrcoef(H[a][:, k], H[b][:, k])[0, 1]) for a in range(3) for b in range(a + 1, 3)]
        per_rubric_hh[r] = round(float(np.mean(vals)), 3)
    summary["humans"]["r_hh_per_rubric"] = per_rubric_hh

    stacked_j, complete = {}, {}
    for jid in judge_ids:
        m = J[jid]
        ok = ~np.isnan(m).any(axis=1)
        complete[jid] = ok
        row = {"n_scored": int(ok.sum())}
        for k, r in enumerate(RUBRICS):
            row[f"sigma_ratio_pool_{r}"] = round(sigma_ratio(m[ok, k], h_pool[:, k]), 3)
            row[f"sigma_ratio_mean_{r}"] = round(sigma_ratio(m[ok, k], h_mean[:, k]), 3)
        row["r95"] = effective_rank_r95(m[ok])
        row["angle_to_meanhuman_deg"] = round(
            largest_principal_angle_deg(stack_z(m[ok]), stack_z(h_mean[ok])), 1)
        row["angle_per_rubric"] = {
            r: round(largest_principal_angle_deg(m[ok, k], h_mean[ok, k]), 1)
            for k, r in enumerate(RUBRICS)}
        summary["judges"][jid] = row

    # correlation families on stacked complete-case scores
    all_ok = np.logical_and.reduce([complete[j] for j in judge_ids])
    jm = np.column_stack([zscore_columns(J[j][all_ok]).reshape(-1) for j in judge_ids])
    hm = np.column_stack([zscore_columns(H[a][all_ok]).reshape(-1) for a in range(3)])
    summary["correlations"] = {k: round(v, 3) if np.isfinite(v) else None
                               for k, v in correlation_families(jm, hm).items()}
    summary["n_items_all_judges"] = int(all_ok.sum())

    # judge-ensemble subspace vs mean-human vector (per-rubric z-scored stacks)
    ens = np.column_stack([stack_z(J[j][all_ok]) for j in judge_ids])
    hv = stack_z(h_mean[all_ok])
    summary["ensemble_angle_to_meanhuman_deg"] = round(largest_principal_angle_deg(ens, hv), 1)
    summary["mean_judge_angle_per_rubric"] = {
        r: round(float(np.mean([summary["judges"][j]["angle_per_rubric"][r] for j in judge_ids])), 1)
        for r in RUBRICS}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# PARIKSHA pilot summary (n={n})\n",
             f"human pair angles (floor): {summary['humans']['pair_angle_deg']}",
             f"human r_hh per rubric: {per_rubric_hh}",
             f"correlations (stacked, n_items={summary['n_items_all_judges']}): {summary['correlations']}",
             f"ensemble angle to mean-human: {summary['ensemble_angle_to_meanhuman_deg']} deg\n",
             "| judge | n | sig_pool L/T/H | r95 | angle |", "|---|---|---|---|---|"]
    for jid, row in summary["judges"].items():
        sig = f"{row['sigma_ratio_pool_ling']}/{row['sigma_ratio_pool_task']}/{row['sigma_ratio_pool_halluc']}"
        lines.append(f"| {jid} | {row['n_scored']} | {sig} | {row['r95']} | {row['angle_to_meanhuman_deg']} |")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    items_file = OUT / "items.csv"
    if cmd == "build":
        items = build_items()
        with open(items_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            w.writeheader()
            w.writerows(items)
        print(f"wrote {len(items)} items -> {items_file}")
        return
    items = list(csv.DictReader(open(items_file, encoding="utf-8")))
    for it in items:
        for i in range(3):
            for r in RUBRICS:
                it[f"h{i+1}_{r}"] = int(it[f"h{i+1}_{r}"])
    if cmd == "judge":
        judge(items)
        judgments = [json.loads(l) for l in open(OUT / "judgments.jsonl", encoding="utf-8")]
        write_html(items, judgments)
    elif cmd == "analyze":
        judgments = [json.loads(l) for l in open(OUT / "judgments.jsonl", encoding="utf-8")]
        analyze(items, judgments)


if __name__ == "__main__":
    main()
