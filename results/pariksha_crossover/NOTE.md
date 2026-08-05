# PARIKSHA crossover — does priority-profile conditioning rotate judge geometry, or only restore variance?

Date: 2026-08-04. Code: `pipeline/pariksha_crossover.py`. Data: `judgments.jsonl` (4,800 rows),
`profiles.json`, `summary.json`. Pre-registration: README "Phase 3 pre-registration → Second
domain (crossover)" bullet (2026-07-14); the 24-priority taxonomy and all analysis choices were
written in the script before any call was made.

## Design in one paragraph

Same 150 PARIKSHA items as the pilot (hindi 60 / telugu 45 / malayalam 45, 3 human raters each),
same multi-rubric single-call prompt and JSON schema (ling 0-2, task 0-2, halluc 0-1), judge =
claude-haiku-4-5. Two conditions, 16 samples/item each: **vanilla16** (unmodified system prompt)
and **conditioned16** (system prompt prefixed with one of 16 fixed evaluative-priority profiles —
Dirichlet(alpha=10) over a 24-priority taxonomy, top-5 renormalized, seed 20260714 — the DMP
mechanism transplanted from Paper A to Paper B's own public data). Both at T=0.7 (documented
divergence from the paper's near-deterministic T=0.2: a 16-sample ensemble needs temperature to
be a fair variance baseline). 4,800 calls total.

## Numbers

| readout | vanilla16 | conditioned16 |
|---|---|---|
| parse rate | 99.9% (2397/2400) | 100.0% (2400/2400) |
| sigma-ratio vs pooled humans (L/T/H) | 1.34 / 1.02 / 1.00 | 1.38 / 1.01 / 1.00 |
| mean within-item std across 16 samples (L/T/H) | 0.035 / 0.065 / 0.072 (mean 0.057) | 0.116 / 0.118 / 0.102 (mean 0.112) |
| ensemble-span angle to mean-human | 53.1° | 52.6° |
| ensemble-MEAN vector angle to mean-human | 54.9° | 54.5° |
| inter-instance r (mean pairwise, 16 instances) | 0.930 [0.89-0.97] | 0.881 [0.83-0.93] |

Human reference on the same 149 complete items: leave-one-out floor 54.8-60.4° (a single rater
vs the mean of the other two); rater-pair angles 58.2-65.1°, rater-pair r 0.42-0.53.

**Matched-variance null** (200 synthetic ensembles = vanilla per-item-rubric mean + independent
gaussian noise with the conditioned condition's per-item-rubric std): mean 53.3°, 95% band
[52.0°, 54.3°]. Observed conditioned span angle **52.6° = 13.5th percentile** of the null —
comfortably inside the band, nowhere near the 2.5th-percentile threshold.

## Verdict on the pre-registered prediction (H3-crossover)

**Confirmed.** Conditioning doubles the within-item spread (0.057 → 0.112 mean per-rubric std;
the "diversity" readout responds exactly as predicted) but the ensemble-span angle to mean-human
does NOT sit below the matched-variance null's 2.5th percentile (13.5th pct), and the unbiased
1-D readout (ensemble-mean angle, 54.9° → 54.5°) barely moves. Pluralistic conditioning on this
bed adds human-flavored item-local noise around the same underlying judgment — variance
restoration without rotation. The 16 "different evaluators" are one judge: inter-instance r only
drops 0.93 → 0.88, still far above the human rater-pair r of ~0.42-0.53.

## Hand-read (8 conditioned transcripts, 3 hindi / 3 telugu / 2 malayalam, varied profiles)

- **7/8 conditioned score triplets are IDENTICAL to the same-seed vanilla sample.** The one
  difference is profile-consistent: a Grammar-pedantry + Error-intolerance profile dropped
  malayalam ling 2 → 1, nitpicking unidiomatic loan-translations.
- No transcript names its priorities or importance scores explicitly. Explanations sometimes
  carry a faint profile flavor (the Register-formality profile praises "appropriately formal
  register", the Terminology-precision profile itemizes precise vocabulary) — but vanilla
  explanations use the same vocabulary, so this is weak evidence of conditioning at best.
- Judgments themselves are competent and grounded (catches a mid-word truncation, a missing
  initial consonant typo, repetitive loops) — the model reads the answers carefully; it just
  doesn't let the profile change its verdict.

## Caveats

- Ceiling/floor context: haiku on PARIKSHA already sits AT the human LOO floor (pilot T=0.2
  angle 56.7° vs floor 54.8-60.4°; here 53-55°), unlike Paper B's headline judges-perpendicular
  result — so there is little rotation headroom on this bed to begin with. The crossover verdict
  is therefore "conditioning does not rotate", not "conditioning fails to close a large gap".
- sigma-ratio vs pooled humans is already ~1 at vanilla T=0.7 (and was ~1 in the T=0.2 pilot):
  on these coarse 0-2 scales the between-item spread was never deficient for haiku; the deficit
  that conditioning repairs is the *within-item* (ensemble) spread, which is why the within-item
  std is the informative spread readout here.
- One judge model, one profile-set seed, 150 items, English-language rubric explanations; the
  null uses gaussian noise on a discrete 0-2 scale (adequate for an angle null, not a generative
  model of scores).
