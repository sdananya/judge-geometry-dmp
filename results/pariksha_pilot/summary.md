# PARIKSHA pilot summary (n=150)

human pair angles (floor): [63.5, 64.9, 58.0]
human r_hh per rubric: {'ling': 0.44, 'task': 0.505, 'halluc': 0.453}
correlations (stacked, n_items=142): {'r_ll': 0.598, 'r_lh': 0.376, 'r_hh': 0.451}
ensemble angle to mean-human: 52.7 deg

| judge | n | sig_pool L/T/H | r95 | angle |
|---|---|---|---|---|
| anthropic/claude-haiku-4-5-20251001 | 150 | 1.325/1.047/1.004 | 3 | 56.7 |
| anthropic/claude-sonnet-4-5 | 150 | 1.315/1.033/1.004 | 3 | 57.6 |
| openai/gpt-4o | 150 | 1.175/0.925/0.925 | 3 | 57.0 |
| openai/gpt-4o-mini | 149 | 1.025/0.756/0.505 | 3 | 64.3 |
| openrouter/google/gemma-3-27b-it | 148 | 0.948/0.875/0.69 | 3 | 64.1 |
| openrouter/meta-llama/llama-3.3-70b-instruct | 146 | 0.707/0.825/0.575 | 2 | 72.5 |
| openrouter/qwen/qwen3-235b-a22b | 148 | 0.95/0.754/0.964 | 3 | 58.5 |