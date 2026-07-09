# Findings so far (Phase 1 pilot, 2026-07-09)

*Written so that someone with no background can follow. Every number sits next to its
baseline, and every claim says which file it comes from. Total paid-API cost: **$0.52**
(the bulk ran on our free Anthropic lane; see `results/spend.jsonl`).*

## The question

When you ask an AI model to judge things people disagree about — moral dilemmas, the
quality of an answer — it behaves like one narrow person: it uses the same few favorite
values over and over, and different AI judges all agree with each other more than any of
them agrees with actual humans.

A recent paper (Paper A) proposed a cheap fix: before each judgment, hand the AI a randomly
drawn "value profile" — e.g. *you care 52% about Responsibility, 20% about Respect for Time,
14% about Consideration…* — so that a crowd of AI judgments looks as diverse as a crowd of
human ones. It reported large improvements.

Our question: **does that fix change what the AI actually judges, or only what its answers
look like?** Think of it this way: if you give one person sixteen costumes, you get sixteen
different-looking judges — but do you get sixteen different *opinions*, or one opinion wearing
sixteen costumes?

## What we did (all of it small-scale, on purpose)

1. Built and unit-tested the measurement tools (20 tests on synthetic data with known
   answers — `metrics/`).
2. Re-scored 150 public items from PARIKSHA (human-rated AI answers in Hindi, Telugu,
   Malayalam; 3 human raters each) with 7 AI judges, to check we can reproduce Paper B's
   "AI judges are geometrically far from humans" diagnostics (`results/pariksha_pilot/`).
3. Ran the actual experiment on 150 moral dilemmas (from the public Scruples corpus, each
   with 40+ real human verdicts): the same AI judge (Claude Haiku) judged every dilemma 16
   times *without* profiles ("vanilla") and 16 times *with* random value profiles ("DMP"),
   ~4,800 judgments total (`results/aita_pilot/`).

## Finding 1 — the "costume" result (the headline)

Giving the judge value profiles changed its *expression* dramatically and its *judgment*
almost not at all:

| what we measured | vanilla | with profiles (DMP) | human reference |
|---|---|---|---|
| share of value-talk concentrated in top-10 values | 0.80 | **0.47** | Paper A: humans 0.35, LLMs 0.82 |
| diversity (entropy) of values in rationales | 0.74 | **0.89** | higher = more diverse |
| does the rationale actually use its assigned profile? | — | **97.6%** | chance ≈ 16–24% |
| disagreement between the 16 judgment samples (std) | 0.032 | **0.088** | humans disagree far more |
| items where all 16 samples gave the same verdict | 92% | 73% | — |
| **correlation of judgments with human verdict rates** | 0.454 | **0.456** | unchanged |
| **mean gap to human verdict distributions** | 0.366 | **0.352** | noise floor 0.085 |

In words: the profiles are visibly *used* (98% of rationales cite them; the judge writes
things like "According to my moral profile, Responsibility (0.517) is my highest value…"),
the answers become much more varied, the value-diversity metrics from Paper A improve almost
exactly as that paper reports — **and the direction of judgment, measured against 40+ real
humans per dilemma, does not move at all** (correlation 0.454 → 0.456). Individual verdicts
do flip under profiles, but the flips are directionless: they don't make the ensemble more
human-like, just noisier.

This is the pre-registered hypothesis H3 (see README): **variance recovers; direction
doesn't.** Pluralistic prompting, at pilot scale, produces pluralism-as-decoration.

Also reproduced independently: Paper A's *diagnosis* is solid — our vanilla judge showed
almost exactly their value-collapse number (0.80 vs their 0.82) despite a different model,
different dilemmas, and a different extraction pipeline. It's their *cure* that appears to
change expression rather than judgment.

## Finding 2 — Paper B's picture reproduces in shape, not in severity

On the public PARIKSHA data (the only part of Paper B's data that is public), with our
pre-registered angle recipe (theirs is unspecified — see caveats):

- **Reproduces:** AI judges agree with each other (r=0.60) much more than they agree with
  humans (r=0.38), and less than humans agree with each other (r=0.45) — the "shared bias"
  signature. Weak judges under-spread and sit outside the human range. On the most
  objective rubric (task quality) judges are *closer* to humans than humans are to each
  other; on the most subjective (hallucination) they fall outside — the paper's
  objective/subjective contrast.
- **Does not reproduce:** the drama. Frontier 2026 judges (Claude Haiku/Sonnet, GPT-4o,
  Qwen-235B) sit at 57–59° from the human consensus — essentially *at* the human-human floor
  (55–60°), nowhere near the near-orthogonal 87–89° the paper reports on its (private)
  headline dataset. Either their private data, their unspecified angle recipe, or their
  older/smaller judges carry that result.

Sanity check that passed before any of this: our human-human agreement computed from raw
public data (r ≈ 0.35) matches the paper's reported ≈ 0.36.

## Things we caught that make us trust the numbers more, not less

1. **Our own first angle computation was wrong** (mixed rubric scales inflated the geometry);
   caught because angles disagreed with correlations, fixed, all downstream numbers from the
   fixed version (`pipeline/run_pariksha_pilot.py`).
2. **The subspace-angle metric is biased toward the noisier condition** — adding variance
   mechanically shrinks the angle of a 16-column ensemble to any target. On a clean subset
   the DMP angle "improved" 51°→44°; the unbiased mean-direction metrics show no movement.
   Phase 2's matched-variance controls (random personas, raised temperature, shuffled
   profiles) exist precisely to kill this artifact, and now we know it's real.
3. **A real confound in our rewriting step:** we rewrite each Reddit dilemma to prevent the
   judge recognizing famous posts (as Paper A did), but 59% of our rewrites left "who is
   being judged" ambiguous — one flipped verdict turned out to be the vanilla judge judging
   the boyfriend while the DMP judge judged the girlfriend. This inflates the absolute gap
   to humans in BOTH conditions (restricting to the 61 unambiguous items improves
   human-correlation from 0.45 to 0.59 in both); it does not affect the vanilla-vs-DMP
   comparison (identical texts in both conditions). Phase 2 fix: rewrites must always name
   the narrator "the main actor."
4. **Paper A's headline improvement did not reproduce here** (they: 22pp→8pp; us: 37pp→35pp),
   and separately, their reported 5pp residual on contested dilemmas is *below the
   mathematical noise floor* (~10pp at ~32 votes/side) implied by their own protocol — one
   of the questions in our author email (`emails/data_requests.md`). Honest caveats on our
   side: smaller judge (Haiku), uniform instead of human-fitted value frequencies, top-5
   profile injection, and the actor-ambiguity noise above.
5. Paper A's "60-value taxonomy" contains 59 values (off-by-one in the paper, verified in
   two independent versions — `pipeline/values_taxonomy.yaml`).

## What this means (if it holds up)

A cheap and popular style of fix — "prompt the judge with diverse personas/values" — makes
diversity *metrics* better without making the judge more *human-aligned*, because the
metrics it improves are exactly the metrics that can't tell opinion from costume. Paper B
showed fine-tuning behaves this way (their Sec 4.8: spread recovers, rotation ≤3%); our
pilot says prompting behaves this way too. If Phase 2 confirms it with proper controls,
the practical message for anyone using LLM judges is: distributional-alignment numbers can
be goodharted by conditioning; per-item directional agreement is the metric that resists it.

## What's next (Phase 2, pending your go)

Controls that make the claim rigorous (random personas, temperature-matched, shuffled-G0
profiles — all pre-registered in README), the fixed rewrite step, a bigger/varied judge set
(is this Haiku-specific?), human-fitted value frequencies for G0, and sending the two author
emails. Pilot cost suggests full Phase 2 stays well under $20 paid.

## Where every number lives

- PARIKSHA: `results/pariksha_pilot/summary.json` (+ `items.csv`, `judgments.jsonl`,
  `transcripts.html`); data provenance `data/PARIKSHA_DATA_NOTE.md`.
- AITA/DMP: `results/aita_pilot/summary.json`, `values_summary.json` (+ `rewrites.jsonl`,
  `judgments.jsonl`, `profiles.json`); data provenance `data/AITA_PILOT_NOTE.md`.
- Spend: `results/spend.jsonl`. Metrics + tests: `metrics/`. Run scripts: `pipeline/`.
