# SIT307 11.1HD — reproducible experiment package

## Study scope
This package reproduces six individual classifiers plus a 5-fold stacking classifier on the supplied `heart.csv`, audits exact duplicate feature records, and evaluates a duplicate-aware redesign across three random seeds.

The selected paper is:
M. Bhagat, A. Sharma and P. Agarwal, “An efficient stacking-based ensemble technique for early heart attack prediction,” *Multimedia Tools and Applications*, vol. 84, pp. 36351–36375, 2025, doi: 10.1007/s11042-024-19293-7.

## Important methodological assumptions
The paper does not explicitly state its train/test proportion, exact sample indices, random state, complete classifier hyperparameters, or a fully reproducible meta-classifier specification. Its published confusion matrices each contain 205 test observations; because 205 is exactly 20% of 1,025 records, the reproduction uses a stratified 80/20 hold-out. Seed 42 and the explicit/default classifier settings in `run_experiments.py` are documented reproduction assumptions, not claims about the authors' exact implementation.

The supplied dataset contains 1,025 rows but only 302 unique full rows / feature profiles. The duplicate-aware protocol removes exact duplicate full rows before train/test splitting, then repeats the evaluation using seeds 42, 7 and 21. Because this changes the evaluation population, the lower duplicate-aware scores should be interpreted as evidence that the conventional row-wise estimate is sensitive to duplicate overlap—not as proof that the published study leaked data, and not as a direct measure of patient-level clinical generalisation. The dataset contains no patient identifier.

The article describes linear-regression imputation and normalisation. The supplied CSV contains no missing values, so imputation is inactive in this reproduction. The code retains explicit median/mode imputers as safeguards, StandardScaler for numeric variables, and one-hot encoding for categorical variables. These implementation choices are disclosed in the report.

## Tested environment
The final student-side run completed successfully on **Python 3.12 (64-bit)** on 28 September 2026. Exact package versions are pinned in `requirements.txt` and listed in `environment_versions.txt`.

## Install and run
Open a terminal in this folder and run:

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 run_experiments.py --data data/heart.csv --output results --seeds 42,7,21
```

On systems where `python` already refers to Python 3.12 64-bit, the equivalent commands are:

```bash
python -m pip install -r requirements.txt
python run_experiments.py --data data/heart.csv --output results --seeds 42,7,21
```

The run should finish with `FINISHED .../results` and regenerate the metric CSVs, ROC plots, confusion matrices and prediction files.

## Expected key outputs
The verified run produced the following stacking accuracies:
- Conventional stratified 80/20 split: **1.0000**
- Duplicate-aware seed 42: **0.8361**
- Duplicate-aware seed 7: **0.8033**
- Duplicate-aware seed 21: **0.8852**

The data audit should report:
- records: **1025**
- unique full rows: **302**
- duplicate full rows: **723**
- conflicting feature labels: **0**

## Package contents
- `data/heart.csv` — dataset used for reproduction.
- `run_experiments.py` — complete executable reproduction and Part 2 experiment.
- `requirements.txt` — pinned dependencies for the final Python 3.12 run.
- `environment_versions.txt` — successful-run environment record.
- `results/all_metrics.csv` — all model/protocol metrics.
- `results/published_table11_transcribed.csv` — five required metrics transcribed from the paper's Table 11.
- `results/paper_comparison.csv` — published versus reproduced baseline comparison.
- `results/data_audit.json` — duplicate audit.
- `results/unique_repeated_summary.csv` — repeated duplicate-aware evaluation.
- `results/*predictions.csv`, `*confusion.csv`, `*roc.png` — stacking evidence.
- `VIDEO_GUIDE.md` — presentation planning notes.

## GenAI acknowledgement
ChatGPT was used for planning, troubleshooting Python/package compatibility, code/documentation editing, and report-structure support. The student executed the final experiment locally, checked the outputs, reviewed the methodology and is responsible for understanding and adapting the final submission in accordance with the unit's academic-integrity requirements.
