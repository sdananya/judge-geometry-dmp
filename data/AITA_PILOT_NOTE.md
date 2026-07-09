# AITA pilot set — 150 moral dilemmas with human judgment distributions

Built 2026-07-09 by `pipeline/build_aita_pilot.py` (seed 20260709). Output: `data/aita_pilot_150.csv`.

## Source + license

**Scruples Anecdotes v1.0** (Lourie, Le Bras & Choi 2020, "Scruples: A Corpus of Community
Ethical Judgments on 32,000 Real-Life Anecdotes", arXiv:2008.09094). Downloaded directly from
the Allen AI release tarball
`https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz`
into `data/raw/scruples/anecdotes/` (gitignored). We did NOT use the HF mirror
(`metaeval/scruples`); the original was tried first and works, and it definitely carries the
full post text. All three splits (train 27,766 / dev 2,500 / test 2,500 = 32,766 posts) were
pooled — the pilot measures judge geometry, it doesn't train on the corpus, so splits don't
matter.

License: the `allenai/scruples` GitHub repo is Apache-2.0; the anecdotes themselves are
public Reddit r/AmItheAsshole posts redistributed by Allen AI (user-generated content —
treat as research-use, don't republish the texts wholesale).

## What a row is

One Reddit AITA post (a real first-person moral dilemma) plus the distribution of community
verdicts. Raw 5-way vote counts come from the corpus's `label_scores`:

- `n_author` — "you (the author) are the asshole" (YTA)
- `n_other` — "the other party is the asshole" (NTA)
- `n_everybody` — "everyone sucks here" (ESH)
- `n_nobody` — "no assholes here" (NAH)
- `n_info` — "not enough info" (recorded, excluded from the binary rate)

Binarization (mirrors Paper A's YTA/NTA collapse and Scruples' own `binarized_label_scores`):

- `n_unacceptable = n_author + n_everybody` (author acted wrongly)
- `n_acceptable = n_other + n_nobody` (author did not act wrongly)
- `n_binary` = their sum; `p_unacceptable = n_unacceptable / n_binary`
- `consensus = max(p_unacceptable, 1 - p_unacceptable)`; `bucket` = its 0.1-wide stratum

Plus `post_id`, `title`, `text` (full post body, HTML-entity-unescaped, otherwise verbatim).

## Selection

1. Filters: `n_binary >= 40` human verdicts; text length 400–4000 chars; non-empty,
   non-`[removed]`/`[deleted]` text. 32,766 → **2,008** qualifying posts
   (pool per bucket: 201 / 255 / 255 / 383 / 914 from lowest to highest consensus).
2. Stratified sample: 30 posts per consensus bucket [0.5-0.6), [0.6-0.7), [0.7-0.8),
   [0.8-0.9), [0.9-1.0], numpy seed 20260709. **Every bucket had >= 30 candidates, so no
   top-up was needed — final counts are exactly 30/30/30/30/30.**

## Sanity numbers

- `n_binary`: min 40, median 75, max 1229, mean 152.8.
- Text length: min 420, mean 1676, max 3920 chars.
- `p_unacceptable` spans 0.00–0.97 (mean 0.40 — AITA verdicts skew "not the asshole",
  matching the corpus-wide skew).
- Schema checks pass: counts sum to `n_binary`, `p`/`consensus` recompute exactly,
  `post_id` unique.
- 10 raw + 5 selected posts (one per bucket) were read end-to-end: all are real, coherent,
  complete first-person dilemmas — no memes, no deleted stubs, none truncated mid-sentence.
  (Corpus-wide only 29/32,766 raw texts are empty; zero carry [removed]/[deleted] markers;
  the filters exclude all of these.)

## Noise-floor implication

Even a PERFECTLY aligned judge won't hit |dP| = 0: both the human rate and a judge rate
estimated from finite samples fluctuate. At p = 0.5 with n_h = n_j = 40 the analytic floor is
sqrt(2/pi) * sqrt(2 * 0.25 / 40) ≈ 0.089 (exact binomial value 0.0889 — verified against
`metrics.geometry.noise_floor_absdiff`).

For THIS pilot, computed exactly with `metrics/geometry.py` (each item's own `n_binary` as
n_h, its observed `p_unacceptable` as the true p, and **n_j = 64** judge samples):

- **mean per-item |dP| noise floor = 0.0513** (range 0.000–0.079 across the 150 items).
- Reference: holding n_h at the filter minimum of 40 gives 0.0622.

So any measured mean |dP| for an LLM judge on this set should be compared against ~0.05,
not 0. Gaps below ~0.05 are indistinguishable from sampling noise at these n; the high-vote
items (median 75, up to 1229 votes) are what pull the floor below the naive 0.089.
