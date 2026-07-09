# Does pluralistic prompting fix LLM judges — or just make them *look* fixed?

Side project (Ananya, started 2026-07-09). Exploratory; candidate PhD SoP direction.

## The question, in plain words

When you ask many humans to judge something subjective (a moral dilemma, the quality of an
answer), you get a *spread* of opinions, and each person's opinions hang together in their own
consistent way. When you ask AI models to judge the same things, two problems show up:

1. **Paper A** ("The Pluralistic Moral Gap", EACL 2026): models mostly voice the same few
   popular values, missing the long tail of human values. A fix called **DMP** (Dynamic Moral
   Profiling) gives each model call a randomly sampled "value profile" (e.g. *care about
   honesty 0.5, loyalty 0.3, fairness 0.2*), and reported ~64% better match to human judgment
   distributions.
2. **Paper B** ("The Geometry of LLM-as-Judge", arXiv:2606.03043): AI judges use less of the
   score range than humans, and — measured geometrically — the *direction* of their scoring
   pattern is nearly perpendicular to humans' (87–89°, where humans sit 78–81° from each
   other). AI judges agree with each other without agreeing with people.

**Our question:** if we apply Paper A's fix (DMP) to Paper B's problem (judge geometry), does
the fix make model judgments *genuinely* more human-like (the scoring direction rotates toward
humans), or does it just add human-flavored randomness (the spread widens but the direction
stays perpendicular)? The second outcome would mean "pluralism as decoration": the numbers that
measure diversity improve while the judgments themselves stay alien.

Paper B already showed that **fine-tuning** produces exactly this decoration effect (spread
recovers fully, rotation toward humans ≤3%). Nobody has tested whether **prompting**
interventions like DMP do the same. That's the gap we're filling.

## Pre-registered hypotheses (written 2026-07-09, BEFORE running Phase 2)

- **H1 (spread):** DMP conditioning raises the judge/human spread ratio σ_J/σ_H substantially
  (expected: from ~0.3–0.8 toward ~1). We expect this almost mechanically.
- **H2 (direction — the real test):** DMP reduces the principal angle between the judge
  ensemble's scoring subspace and the human one. This is what "genuine pluralism" requires.
- **H3 (dissociation):** spread recovery and direction change come apart. **Quantitative
  prediction:** spread recovers ≥50% of the gap toward σ_J/σ_H = 1, while the
  rotate-toward-human share of the score change (Paper B's Sec-4.8 decomposition) stays ≤10%.
- **H4 (conditional):** per-item alignment with human judgment distributions on *contested*
  items improves beyond the sampling-noise floor **only if** H2 holds.

Controls that make H3 testable (pre-committed): (a) random non-value personas with matched
token count; (b) temperature increase with matched sample count; (c) shuffled-G0 profiles
(same Dirichlet machinery, value frequencies permuted — keeps the diversity mechanics, destroys
the human-informed content).

**Pre-registered metric operationalizations** are in `metrics/geometry.py` (Paper B never
specifies its principal-angle recipe; ours is fixed here before any experimental comparison:
center per-column z-scores across items, subspace = span of top-k right singular vectors /
score columns, angles via SVD of Q_Aᵀ·Q_B per Björck & Golub 1973).

**Known measurement trap (pre-registered):** with ~32 judgments per side, the per-item
|ΔP(acceptable)| metric has a noise floor of ~0.10 at 50/50 items even under PERFECT alignment.
Every distributional result must be reported as excess above this analytic floor
(`metrics.noise_floor_absdiff`). Paper A's reported 5pp low-consensus residual is *below* this
floor — one of the things Phase 1 checks.

## Layout

- `metrics/` — the two papers' metric families, unit-tested on synthetic data with known geometry
- `pipeline/` — judge runners (vanilla / DMP / controls)
- `data/` — data notes; `data/raw/` (gitignored) holds clones/downloads
- `results/` — persisted run outputs (raw generations + per-row labels + summaries)
- `emails/` — data-request drafts to both papers' authors
- `REPORT.md` — running findings, written in plain language

## Ground rules

Frugality: Anthropic API + SLURM box are the workhorses; OpenAI/OpenRouter only for small
anchor-judge runs (log every paid call's cost in REPORT.md). Simplest thing first: look at raw
data before computing on it; hand-read transcripts before trusting judges; in-distribution
sanity checks before novel claims. One epoch of skepticism per surprising number.
