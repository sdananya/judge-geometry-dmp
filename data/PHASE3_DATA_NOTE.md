# Phase 3 data — 500-item set, rewrites, topics, per-topic G0, few-shot pool

Built 2026-08-04 by `pipeline/build_phase3_data.py` (seed **20260714** throughout), per the
Phase 3 pre-registration in README.md. Not committed to git per instruction.

## 1. `data/aita_pilot_500.csv`

Same schema as `aita_pilot_150.csv`. All 150 existing rows kept (byte-identical values,
verified per-row), each consensus bucket topped up to **100 items** (+70/bucket) from the
qualifying Scruples pool (same filters: n_binary >= 40, 400-4000 chars, non-deleted;
pool = 2,008). Candidates available per bucket after excluding the 150:
171 / 225 / 225 / 353 / 884 (0.5-0.6 → 0.9-1.0). Verified: 500 rows, exactly 100/bucket,
all 150 old post_ids present.

## 2. `results/phase3/rewrites.jsonl` + `rewrite_gate.json`

Schema: `{post_id, rewrite, anchored, mem_flagged}`, one row per pilot item in CSV order.
The 150 phase-2 rows are **copied verbatim** (byte-identical lines, verified 150/150);
the 350 new items were rewritten with gpt-4o-mini + `REWRITE_SYS_V2` (temp 0.3,
max_tokens 500) and probed 3x for memorization (`MEM_SYS`/`MEM_USER`, temp 1).

Gate (over all 500): **anchored_frac = 1.000 (gate_ok)** · **mem_flagged = 13/500**
(4 old + 9 new) · **fabricated_demo_frac = 0.020** (10/500).

**Fabricated-demo caveat (hand-checked, all 10 flags):** every flagged rewrite's age IS in
the original — the original-marker regex just misses abbreviation forms
("15y old", "11 yo", "7yr", "17 years", bare "16" for a dog's age). Inspected true
fabrication rate among flags: **0/10**; 0.020 is a loose upper bound from a conservative
regex, not evidence of invention.

Hand-read of 5 new rewrites (a8ho5t, ap6i55, b93wog, alns7z, ay0oc2): all anchored on
"the main actor", faithful to the morally relevant facts and event order, no invented
demographics, sensitive context preserved where relevant (e.g. a8ho5t keeps the sister's
sexual orientation + adopted nephew). Minor stylistic notes: rewrites of short posts run
slightly longer than the originals; occasional mild euphemism at the end
("unkind person" for "asshole"), which is intended AITA-deframing.

## 3. `data/topics_500.json`

`{post_id: topic}` for all 500, labeled by claude-haiku-4-5-20251001 (temp 0, input = the
REWRITE text) into 8 fixed labels. 0 parse retries, 0 fallbacks — every first call
returned a verbatim label.

| topic | n | frac |
|---|---|---|
| romantic_relationships | 212 | .424 |
| family | 81 | .162 |
| friendship_social | 54 | .108 |
| work | 47 | .094 |
| money_property | 40 | .080 |
| household_neighbors | 28 | .056 |
| events_celebrations | 25 | .050 |
| other | 13 | .026 |

## 4. `data/g0_topics.json`

Per-topic value priors from the EXISTING mattboraske extractions
(`g0_value_mentions.jsonl`, 1,500 comments / 3,510 mentions, 316 distinct source posts).
Each source post labeled with the same 8 topics by haiku (temp 0, parquet title+text
truncated to 1,500 chars; 0 retries/fallbacks). G0_t = add-one-smoothed value frequency
over the 59 taxonomy values within the topic's comments; keys exactly match the taxonomy,
each distribution sums to 1. Topics with <100 mentions fall back to the global
distribution (marked in `meta.fallback_topics_lt100`).

Mentions per topic: family 1,295 · romantic_relationships 790 · work 333 ·
friendship_social 307 · money_property 278 · household_neighbors 257 ·
events_celebrations 250 · **other 0 → FELL BACK to global** (no post labeled "other").

Face validity of the conditioning: money_property tops out at Responsibility/.160 +
Property/.092; household_neighbors surfaces Safety/.092; everything else is led by
Respect/Autonomy as in the global G0.

## 5. `data/fewshot_pool.json`

8 elicitation examples, seed 20260714, drawn from the qualifying pool EXCLUDING the 500
(verified disjoint): one each from 0.5-0.6 / 0.6-0.7 / 0.7-0.8, two from 0.8-0.9, three
from 0.9-1.0 with mixed majorities (pct_unacceptable 0 and 6 = acceptable-majority,
93 = unacceptable-majority). Rewritten with `REWRITE_SYS_V2` (gpt-4o-mini); **all 8 pass
the anchor check on the first draw** (no replacements needed). Fields:
`{post_id, bucket, p_unacceptable, rewrite, pct_unacceptable}` with
`pct_unacceptable = round(100 * human p_unacceptable)`.

## Spend

All rewrites/probes on gpt-4o-mini: 1,408 paid calls this build, ~483k prompt + ~77k
completion tokens ≈ **$0.12** (logged in `results/spend.jsonl`). Topic labeling
(500 + 316 calls) on claude-haiku via the free lane.
