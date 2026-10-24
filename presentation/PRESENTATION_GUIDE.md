# Ten-minute LecTrace presentation guide

Open the deployed LecTrace page, press `F` for fullscreen, and use the right arrow to advance. The script contains exactly five computational states.

| Time | Section | Speaking focus |
|---:|---|---|
| 0:00–1:10 | Healthcare problem | A source-domain score may not transfer across hospitals; accuracy alone cannot reveal shortcuts. |
| 1:10–2:30 | Question and design | PneumoniaMNIST source, RSNA external classification and boxes, one-shot CheXpert confirmation. |
| 2:30–3:35 | State 1: resolution | Hold the model family fixed. B64 retains 5.22 times as many pixels and gives the largest accuracy gain. |
| 3:35–4:40 | State 2: forward pass | Explain features, two logits, softmax probability, margin, and the difference between ranking and a fixed threshold. |
| 4:40–6:20 | State 3: AC-JSD | Compare three views, step into `jensen_shannon`, and explain why disagreement is penalized. |
| 6:20–7:45 | State 4: external transfer | RSNA AUROC improves clearly; RSNA accuracy changes little. CheXpert gives a positive final one-shot result. |
| 7:45–9:10 | State 5: spatial evidence | Step through area-corrected energy. AC-JSD improves over plain, Aug1 aligns best, and the center prior remains stronger. |
| 9:10–10:00 | Conclusion | Resolution drives accuracy; consistency improves external ranking; spatial evidence improves modestly and is not causal proof. |

## Navigation

- `→`: next execution step
- `←`: previous execution step
- `Shift+→`: step over a helper call
- `u`: return from a helper function
- `E`: show or hide variables
- `R`: switch between rendered content and source code
- `F`: fullscreen

The single-case logits and view probabilities are compact teaching examples used to expose the calculations. All cohort-level tables, confidence intervals and conclusions are recorded outputs from the complete experiment notebook.
