| cond | corr | |dP| | excess | w_std | unanim | inst_r | angle | lowcons |
|---|---|---|---|---|---|---|---|---|
| vanilla | 0.465 | 0.3581 | 0.2735 | 0.024 | 0.94 | 0.957 | 59.0 | 0.487 |
| dmp_empirical | 0.455 | 0.355 | 0.2703 | 0.079 | 0.787 | 0.867 | 55.2 | 0.46 |
| dmp_uniform | 0.46 | 0.3467 | 0.2621 | 0.083 | 0.78 | 0.86 | 54.0 | 0.461 |
| shuffled_g0 | 0.476 | 0.343 | 0.2584 | 0.085 | 0.773 | 0.859 | 55.5 | 0.454 |
| random_persona | 0.506 | 0.3459 | 0.2613 | 0.058 | 0.853 | 0.898 | 53.3 | 0.465 |
| diversity_instruction | 0.416 | 0.3784 | 0.2938 | 0.012 | 0.967 | 0.98 | 61.9 | 0.498 |

bootstrap deltas (95% CI):
  dmp_empirical_minus_vanilla: corr -0.0091 CI[-0.075, 0.0491], |dP| -0.0036 CI[-0.0272, 0.0212]
  dmp_empirical_minus_random_persona: corr -0.0517 CI[-0.1037, -0.0074], |dP| 0.0092 CI[-0.0069, 0.0266]
  dmp_empirical_minus_shuffled_g0: corr -0.0214 CI[-0.0613, 0.0145], |dP| 0.0123 CI[-0.0025, 0.0267]
  dmp_empirical_minus_diversity_instruction: corr 0.0407 CI[-0.0231, 0.1074], |dP| -0.0239 CI[-0.0511, 0.001]
  dmp_uniform_minus_vanilla: corr -0.0039 CI[-0.0707, 0.0558], |dP| -0.0118 CI[-0.0348, 0.0131]