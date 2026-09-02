# Reference corpus

15 arXiv preprints, all submitted before 2023-01-01, spanning computer
science and statistics, used to calibrate TextScope's style suggestions
against real published scientific writing (see `README.md` — "Calibration").

Built with `cli.py calibrate reference_corpus/txt/*.txt --lang en --out reference.json`.

PDFs were fetched from arXiv and converted with `pdftotext`; the
bibliography/references section was stripped (heuristically — see the
tail of each .txt if you need to double check) since a reference list is
not prose and would skew the stylometric averages. `pdf/` is not tracked
in git (binaries); `txt/` is, so the corpus can be reproduced without
re-downloading.

| arXiv ID   | Category | Title                                                                 | Submitted  |
|------------|----------|------------------------------------------------------------------------|------------|
| 2212.14542 | cs.DC    | Recurrent Problems in the LOCAL Model                                  | 2022-12-30 |
| 2212.14604 | cs.SE    | Testing RESTful APIs: A Survey                                         | 2022-12-30 |
| 2212.14651 | cs.DC    | Anticipation of Method Execution in Mixed Consistency Systems          | 2022-12-30 |
| 2212.14777 | stat.ME  | Polynomial Spline Regression: Theory and Application                   | 2022-12-30 |
| 2301.00057 | cs.SE    | A Mapping of Assurance Techniques for Learning Enabled Autonomous Systems | 2022-12-31 |
| 2301.00068 | cs.CL    | Inconsistencies in Masked Language Models                              | 2022-12-30 |
| 2301.00077 | stat.AP  | A Study on a User-Controlled Radial Tour for Variable Importance        | 2022-12-31 |
| 2301.00241 | stat.ML  | Contextual Bandits and Optimistically Universal Learning                | 2022-12-31 |
| 2301.00254 | cs.CV    | Depression Diagnosis and Analysis via Multimodal Multi-order Factor Fusion | 2022-12-31 |
| 2301.00265 | cs.CV    | Source-Free Unsupervised Domain Adaptation: A Survey                    | 2022-12-31 |
| 2301.00270 | cs.LG    | NetEffect: Discovery and Exploitation of Generalized Network Effects    | 2022-12-31 |
| 2301.00280 | cs.AI    | RECOMED: A Comprehensive Pharmaceutical Recommendation System           | 2022-12-31 |
| 2301.00301 | cs.LG    | Generalized PTR: User-Friendly Recipes for Data-Adaptive Algorithms with Differential Privacy | 2022-12-31 |
| 2301.00303 | cs.CL    | Rethinking with Retrieval: Faithful Large Language Model Inference       | 2022-12-31 |
| 2301.01602 | cs.AI    | Unpacking the "Black Box" of AI in Education                            | 2022-12-31 |

Selected via the arXiv API (`export.arxiv.org/api/query`), sorted by
submission date descending within each category, filtered to
`submittedDate:[20100101 TO 20221231]`, then picked for category spread
(all of cs.AI, cs.CL, cs.CV, cs.DC, cs.LG, cs.SE, stat.AP, stat.ME,
stat.ML). No cherry-picking by content — titles were not read before
selection.
