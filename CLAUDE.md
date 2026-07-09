# judge-geometry-dmp — working notes for Claude sessions

Side project (Ananya, MATS; started 2026-07-09). Separate from `prompted-false-facts` — do
NOT apply that repo's SLURM/box/HF sync rules here unless GPU work actually appears.

## What this is

Testing whether DMP-style pluralistic prompting (Paper A: "The Pluralistic Moral Gap",
EACL 2026, aclanthology 2026.eacl-long.305 / arXiv 2507.17216) genuinely rotates LLM-judge
scoring geometry toward humans, or only restores score variance (Paper B: "The Geometry of
LLM-as-Judge", arXiv 2606.03043). Pre-registered hypotheses H1–H4: README.md. Running
findings: REPORT.md. Full verified paper details: the project memory in
`~/.claude/projects/-Users-Ananya-anthropic-fellowship-projects-prompted-false-facts/memory/sideproject-judge-geometry-dmp.md`
(written before this repo existed).

## Key facts a fresh session must know

- **Neither paper released data/code.** Only public per-annotator resource: PARIKSHA
  (github microsoft/PARIKSHA + karya-inc/pariksha), cloned in `data/raw/` (gitignored).
  Gotchas (swap bug, loader rules): `data/PARIKSHA_DATA_NOTE.md`.
- **AITA side**: Scruples Anecdotes (Allen AI, Apache-2.0) → `data/aita_pilot_150.csv`
  (150 posts, ≥40 binary verdicts each, 5 consensus buckets × 30). See `data/AITA_PILOT_NOTE.md`.
- **Paper A's Table 5 lists 59 values, not 60** (off-by-one in the paper): `pipeline/values_taxonomy.yaml`.
- Paper B's principal-angle recipe is UNSPECIFIED in the paper; ours is pre-registered in
  `metrics/geometry.py` (z-score per rubric before stacking — raw stacking inflates angles).
- Paper B Sec 4.8 already shows fine-tuning = stretch-without-rotation; our novelty is the
  PROMPTING analogue.

## Practices

- Secrets in `.env` (gitignored, mode 600): OPENAI_API_KEY, OPENROUTER_KEY. Anthropic keys
  come from the shell env (LP key preferred for bulk runs). The box copy of OPENROUTER_KEY
  (node-18 /workspace-vast/sdananya/.env) has a broken trailing `=` — local copy is fixed.
- Frugality: Anthropic = workhorse; OpenAI/OpenRouter only for anchor judges. Every paid
  call auto-logs to `results/spend.jsonl` via `pipeline/llm.py` (disk-cached — reruns free).
- Simplest-first; read raw data and transcripts before trusting aggregates; every rate next
  to its baseline/noise floor (`metrics.geometry.noise_floor_absdiff`); surprising numbers
  are bugs until proven otherwise.
- Findings written in plain language (naive-reader-friendly) in REPORT.md.
- venv: `.venv` (uv; numpy pandas httpx pytest). Tests: `cd metrics && ../.venv/bin/python -m pytest -q`.

## Pipeline map

- `pipeline/run_pariksha_pilot.py` — build / judge / analyze (Step 2 reproduction; done,
  results in `results/pariksha_pilot/`)
- `pipeline/run_aita_pilot.py` — rewrite / judge / analyze / values / analyze_values
  (Step 3 DMP pilot; documented divergences from Paper A in its docstring)
- `emails/data_requests.md` — author data-request drafts (Step 0; not yet sent as of 2026-07-09)
