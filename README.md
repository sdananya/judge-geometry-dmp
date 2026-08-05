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

## Phase 2 pre-registration amendments (written 2026-07-14, BEFORE the Phase 2 runs)

- **Conditions (150 dilemmas × 16 samples each, Claude Haiku, T=1):** vanilla /
  dmp_empirical (Dirichlet(10·G0) with G0 fitted to values extracted from real AITA
  commenter rationales) / dmp_uniform (pilot condition, G0 sensitivity) / shuffled_g0
  (G0_emp permuted across values — keeps diversity mechanics, destroys human-informed
  content) / random_persona (non-value personas, format- and token-matched to DMP
  profiles) / diversity_instruction (vanilla + "reasonable people disagree; voice one
  plausible human opinion").
- **Deviation from original pre-registration:** the planned temperature-increase control is
  infeasible — the Anthropic API caps temperature at 1.0 and vanilla already runs there.
  The diversity_instruction condition replaces it as the non-value variance-inducer.
- **Primary metric (H2): correlation between per-item mean judgment and human verdict
  rates.** The ensemble span-angle is REPORTED but only interpreted against the
  variance-matched controls (Phase 1 showed it mechanically favors noisier conditions).
- **Inference:** paired item bootstrap, B=1000, 95% percentile CIs on condition deltas
  (dmp_empirical − vanilla; dmp_empirical − random_persona; dmp_empirical − shuffled_g0).
- **H3 verdict rule:** H3 is supported if dmp_empirical raises within-item variance by
  ≥2× over vanilla while the primary correlation's 95% CI on (dmp − vanilla) includes 0;
  genuine pluralism requires the correlation delta CI to exclude 0 AND exceed the
  matched-variance controls' deltas.
- **Rewrite fix:** the rewriter must always name the narrator "the main actor"; runs are
  gated on ≥95% of rewrites passing the string check plus a 5-item hand-read.

## Phase 3 pre-registration (written 2026-07-14, BEFORE the Phase 3 runs) — paper must-haves

- **Scale & equivalence:** 500 dilemmas (100/consensus-bucket, same Scruples filters, same
  rewrite gate). Negative claims use TOST equivalence bounds pre-set at **δ = 0.05
  correlation points** (paired item bootstrap; "no effect" = 95% CI within ±δ). Profile-set
  sensitivity: dmp_empirical re-run with 2 extra profile-set seeds on the 150-item subset;
  claim requires consistency across seeds.
- **Faithful-DMP condition** (Paper A protocol as close as reconstructable): profiles
  resampled PER dilemma (not fixed), N = 32 samples/dilemma (their judgment count),
  topic-conditioned G_t ~ Dirichlet(10·G0_t); topics assigned by LLM labeling into ~8
  categories (their method is unspecified — documented proxy), G0_t fitted per topic from
  the MattBoraske human-rationale extractions. Council baseline = pooled samples across all
  vanilla judges, N matched.
- **New baselines:** (a) **distribution elicitation** — ask the judge directly for the % of
  people who would judge unacceptable (3 samples, mean); (b) **few-shot elicitation** — same
  with 8 in-context (dilemma → human %) examples from non-overlapping items; (c)
  **temperature control** on OpenAI judges at T=1.5 (infeasible on Anthropic, documented).
- **Judge roster:** full grid on claude-haiku (free); key conditions (vanilla, dmp_empirical,
  random_persona, elicitation) on claude-sonnet, gpt-4o-mini, gpt-4o, llama-3.3-70b,
  qwen3-235b, gemma-3-27b. Paid budget cap for all of Phase 3: **$100**; log in spend.jsonl.
- **Decomposition:** Paper B Sec-4.8 stretch/rotate/residual applied to every condition's
  per-item delta vs vanilla (base = vanilla direction, human = verdict-rate vector);
  headline = rotation share, comparable to their fine-tuning ≤3%.
- **Second domain (crossover):** PARIKSHA 150-item bed; judges conditioned with 16
  evaluative-priorities profiles (fluency-first / strictness / cultural-context / etc. —
  taxonomy written before running); metrics = σ-ratio, r95, angle to mean-human vs the
  3-rater human floor, WITH a matched-variance null (synthetic columns preserving per-item
  mean/variance). Prediction (H3-crossover): σ recovers, angle does not beat the null.
- **Primary metric unchanged** (correlation of per-item means with human rates); all |ΔP|
  vs noise floor; every rate with CIs.

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
