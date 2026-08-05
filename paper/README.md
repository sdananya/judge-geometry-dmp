# Paper draft — NeurIPS format

Compile: upload this folder to Overleaf (pdfLaTeX): main.tex + neurips_2024.sty + refs.bib + figs/.
Currently `[preprint]` mode; switch the usepackage option for submission (anonymous) or camera-ready (`final`).

Before submitting:
- TODO markers in main.tex: author block, repo link, and the three data-freeze items
  (qwen3-235b + gemma-3-27b judges, T=1.5 control) — fold in from results/phase3/summary.json when the stream finishes.
- VERIFY first names in refs.bib for russo2026pluralistic and mukherjee2026geometry (guessed from common usage; check the PDFs).
- Update to the current year's NeurIPS style file (2024 vendored copy used for drafting).
- Number pass: every number in the text against results/phase2/summary.json, results/phase3/summary.json,
  results/pariksha_crossover/summary.json, results/*/values_summary.json.
