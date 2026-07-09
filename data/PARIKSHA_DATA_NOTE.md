# PARIKSHA human-eval data — inspection note

Date: 2026-07-09. Data lives in `data/raw/` (gitignored). Sources cloned shallow:

- `data/raw/pariksha-microsoft/` — https://github.com/microsoft/PARIKSHA (renamed dir; can't use
  `PARIKSHA` + `pariksha` side by side on the Mac's case-insensitive filesystem)
- `data/raw/pariksha-karya/` — https://github.com/karya-inc/pariksha

Paper: Watts et al. 2024, "PARIKSHA: A Large-Scale Investigation of Human-LLM Evaluator Agreement
on Multilingual and Multi-Cultural Data" (arXiv:2406.15053).

## Verdict up front

**Re-judging with our own LLM judges is fully feasible.** The Microsoft repo has, for every rated
item: question text + system prompt (`outputs/round1/formatted/output_<lang>.json`), the full model
answer text, per-rater human labels **with rater identity** (`worker_id`), per-rater free-text
justifications in the native language, and GPT-4 judge scores for the same three rubrics
(`gpt_eval/outputs/round1/individual/`). Join is exact and 100% complete (after fixing one
label-swap bug, below).

## The files

### 1. Human ratings (primary): `pariksha-microsoft/karya_eval/round1/individual/<lang>_ratings_output_indi[_just].tsv`

One TSV per language (10 languages), one **row per (model, prompt, rater)** — 6,285 rows total.
Columns (identical across files; `_just` vs non-`_just` filenames have the same 9-column schema):

```
model | prompt_id | worker_id | ling_rating | task_rating | hallucination_rating
| ling_rating_reasoning | task_rating_reasoning | hallucination_reasoning
```

- `ling_rating` (linguistic acceptability): {0,1,2}. Distribution 0:467 / 1:1791 / 2:4027.
- `task_rating` (task quality): {0,1,2}. Distribution 0:1661 / 1:2048 / 2:2576.
- `hallucination_rating`: {True,False}, **True = hallucination present** (confirmed by reading
  justifications). 45% True overall — high, but the low-end open models really are that bad.
- `worker_id`: **YES, individual rater identity is preserved.** 30 distinct workers overall,
  exactly 3 per language (malayalam has 4; one worker contributed only a handful of rows).
  Reasoning columns are populated for 6/10 languages (hindi, gujarati↔punjabi, malayalam, marathi,
  tamil have them; bengali/kannada/odia/telugu have rating-only rows).

### 2. Model answers: `pariksha-microsoft/outputs/round1/formatted/output_<lang>.json`

List of 20 prompt objects per language. Each object: `prompt_id`, `prompt_timestamp`,
`prompt_creator`, `prompt_type` (domain), `language`, `system_prompt`, `prompt` (question text),
then **one key per model** (e.g. `"gpt-4": {"response": ..., "timestamp": ...}`). 10–20 models per
language (Hindi has 20; incl. GPT4o, gpt-4, gpt-35-turbo, gemini-pro, Llama-3-70B/8B, aya-23-35B,
gemma-7b-it, Mistral-7B, and many Indic community models like Airavata, Navarasa, Gajendra,
language-specific llamas).

**Join keys:** human TSV `(prompt_id, model)` → answer JSON `(prompt_id, model-key)`, within
language. After the swap fix: **1,976/1,976 rated (model,prompt) items join to a non-empty answer
(100%, zero empty responses).** Answer length: median 572 chars, p10 214, p90 1440.

### 3. GPT-judge scores (for later comparison): `pariksha-microsoft/gpt_eval/outputs/round1/individual/{task_quality,linguistic_acceptability,hallucinations}/individual_<lang>.json`

Same 20-prompt structure; each model key → `{"score": int, "justification": <english text>}`.
Same join keys.

### 4. karya-inc repo: `pariksha-karya/pariksha-individual.csv` (2,880 rows) and `pariksha-pair.csv`

Individual CSV: one row per (prompt, model) with columns `prompt_id, prompt_type, language, prompt,
human_{1,2,3}_{LA,TQ,H}, human_{1,2,3}_{LA,TQ,H}_just, human_LA, human_TQ, human_H, human_score,
all_human_agree_{H,LA,TQ}`. Raters are anonymised to slots 1/2/3 (no worker IDs) and — critically —
**there is NO model column and NO answer text**, so rows can't be attributed to a model. All 200
prompt_ids overlap with the Microsoft repo. Use this repo only as a cross-check (its language
labels are correct, which is how we confirmed the bug below); the Microsoft repo is the primary
source.

## BUG FOUND — gujarati/punjabi TSVs are swapped

`karya_eval/round1/individual/gujarati_ratings_output_indi_just.tsv` actually contains **Punjabi**
data (its prompt_ids match `output_punjabi.json`, karya-inc labels those prompts punjabi, and its
justifications are in Gurmukhi script), and the `punjabi_...tsv` file contains **Gujarati** data
(Gujarati script justifications, gujarati prompt_ids). Before the fix, those two languages showed
0% answer join. **Any loader must relabel: gujarati file → punjabi, punjabi file → gujarati.**
(The pairwise TSVs were not checked for the same bug — re-verify before using them.)

## Counts (after swap fix)

| language | rating rows | models | workers | 3-rater items (with answers) | median answer chars |
|---|---|---|---|---|---|
| hindi | 1132 | 20 | 3 | 360 | 841 |
| kannada | 707 | 14 | 3 | 147 | 472 |
| telugu | 652 | 12 | 3 | 205 | 427 |
| tamil | 640 | 14 | 3 | 181 | 563 |
| bengali | 630 | 15 | 3 | 175 | 533 |
| punjabi | 599 | 13 | 3 | 178 | 316 |
| malayalam | 581 | 14 | 4 | 169 | 537 |
| marathi | 514 | 12 | 3 | 132 | 567 |
| gujarati | 437 | 10 | 3 | 106 | 381 |
| odia | 393 | 12 | 3 | 84 | 342 |

- Distinct (language, model, prompt) items: **2,371**. With **>=2 raters: 2,174**; with
  **>=3 raters: 1,737** (+3 items with 4 raters, in malayalam). 20 prompts per language.
- Domains (`prompt_type`, from the output JSONs; 3-rater item counts): **culture 714, finance 474,
  health 430**; kannada uses `cultural`(54)/`factual`(24) instead of `culture`, tamil has
  `creative`(19)/`factual`(22). Note: the geometry paper's domain names/counts (health & wellness
  284, finance 323, everyday life 575) do not match these exactly — likely a relabelled/filtered
  subset ("everyday life" ≈ culture?); don't expect to reproduce their n's exactly.

## Human–human agreement floor (pairwise Pearson, stacked over all rater pairs within language; n = 5,657 pairs)

| rubric | stacked r | per-language range |
|---|---|---|
| linguistic acceptability {0,1,2} | **0.346** | odia 0.14 … kannada 0.48 (hindi 0.42) |
| task quality {0,1,2} | **0.512** | odia 0.10 … punjabi/marathi 0.66 (hindi 0.47) |
| hallucination {0,1} | **0.350** | odia −0.01 … marathi 0.55 (hindi 0.41) |

This matches the paper's reported ≈0.36 stacked human–human Pearson for LA and hallucination
almost exactly; task quality is notably easier (0.51). **Odia is an outlier with near-zero
inter-rater agreement on two rubrics — exclude it from any agreement-floor analysis.**
Unanimity among the 1,737 three-rater items: LA 56%, TQ 42%, hallucination 54% unanimous.

## Impressions from 10 read examples (question + answer + 3 ratings each)

Read 2 each from hindi, bengali, telugu, malayalam, punjabi (all with 3-rater overlap).

- Data looks real and clean: questions are natural everyday/cultural/health/finance questions in
  the native script ("how do I save money on daily expenses?", "which Guru bore the Miri-Piri
  swords?"); answers are genuine LLM outputs of 200–3,500 chars, correct encoding, no truncation
  or empty fields anywhere (0 empty responses).
- Quality spread is wide and the ratings track it: a degenerate Llama-2-7b Hindi answer (Hinglish
  preamble "Certainly, I'd be happy to help!", then repetitive loops) got LA=1, TQ∈{0,1},
  H=True from all 3 raters; a clean gemini-pro Hindi answer got straight 2/2/False from all 3;
  a repetitive malayalam-llama answer (repeats "temples" four times in one list) still got
  H=False but LA/TQ=1 from 2 of 3 raters.
- Weaker models often loop or parrot the question back; the strong closed models (GPT4o, gemini)
  read fluently. Some answers open with English boilerplate before switching to the target
  language — raters seem to penalise this under LA.
- Disagreement is visibly subjective: e.g. a bengali vitamin-D answer got (LA=1,H=False),
  (LA=1,H=False), (LA=0,H=True) — the same mildly-circular answer read as harmless by two raters
  and hallucinated by the third. Hallucination is clearly the noisiest label.
- Rater justifications are speech-to-text-ish native-language prose (spelling of "hallucination"
  varies wildly: हेलुशेनेंस/हेल्यूजंस/એલુસીનેશન), often generic ("no mistakes, so I chose 2").
  Present only for some languages; fine as color, not as data.

## Recommended 150-item pilot subset

Restrict to items with **all 3 raters** and answer text (1,737 available). Take:

- **Hindi 60** (high-resource; 360 candidates; 20 models incl. GPT4o/gpt-4/gemini for a quality
  spread) — stratify ~30 culture / 15 finance / 15 health, and spread across models (max 3–4 items
  per model).
- **Telugu 45** (lower-resource, decent agreement floor 0.25–0.42, 205 candidates).
- **Malayalam 45** (lower-resource, the best hallucination agreement 0.49, 169 candidates,
  justifications present).

Same stratification for the other two languages. Avoid odia (broken agreement floor) and the
kannada/tamil files if domain comparability matters (nonstandard prompt_type values). Sampling unit
= (language, model, prompt_id); each item carries 3 human scores per rubric plus GPT-4 scores for
free.

## Gotchas checklist for the loader

1. Swap gujarati↔punjabi TSV language labels (see BUG above).
2. `hallucination_rating` is the string "True"/"False" → map True→1 (hallucination present).
3. ~197 items have only 1 rater and 437 have 2 — filter to >=3 for the floor.
4. Model keys in output JSONs match the TSV `model` column verbatim (incl. slashes, e.g.
   `ai4bharat/Airavata`); join within language on (prompt_id, model).
5. karya-inc CSV has no model column — don't try to use it for re-judging.
