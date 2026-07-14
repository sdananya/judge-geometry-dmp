# G0: empirical base measure over the 59 moral values

**What it is.** A prior distribution G0(v) over the 59-value taxonomy
(`pipeline/values_taxonomy.yaml`, Russo et al. EACL 2026 Table 5), estimated from real
human moral rationales — mirroring Paper A, which set G0(v) = frequency of value v across
human rationales.

**How it was built.** From the HF dataset
`MattBoraske/reddit-AITA-submissions-and-comments-multiclass` (test split),
1500 top-level comments were sampled (seed 20260714; up to 5 per post,
classification YTA/NTA only, 80-1200 chars, bot/AutoModerator text dropped).
Each comment went to `claude-haiku-4-5-20251001` (temp 0) with the same VALUES_SYS
instruction as the AITA pilot: return at most 3 values from the fixed list that the
rationale clearly relies on. This produced 3510 value mentions
(302 comments yielded none). G0(v) = mention count of v / total mentions, with
**add-one smoothing** over mention counts before normalizing so all 59 values get
nonzero mass (6 values had zero raw mentions).

**Top-10 values (raw, pre-smoothing frequencies):**

| value | mentions | freq |
|---|---|---|
| Respect | 557 | 0.159 |
| Autonomy | 410 | 0.117 |
| Responsibility | 288 | 0.082 |
| Consideration | 285 | 0.081 |
| Compassion | 208 | 0.059 |
| Justice | 185 | 0.053 |
| Integrity | 176 | 0.050 |
| Parental Responsibility | 147 | 0.042 |
| Care | 136 | 0.039 |
| Pragmatism | 134 | 0.038 |

**Concentration.** Top-10 mass = 0.720 (raw); normalized entropy = 0.756
(smoothed; 0.766 on raw counts). Uniform over 59 values would be 1/59 = 0.0169
per value. Paper A's human reference: top-10 mass ~0.352 (humans) vs ~0.816 (LLMs) —
this G0 is on the human side of that gap by construction.

File: `data/g0_empirical.json` (probs sum to 1 over all 59 values). Not committed
(data/raw is gitignored; these two files are small derived artifacts).
