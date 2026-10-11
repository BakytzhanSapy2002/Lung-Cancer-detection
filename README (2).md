# Lung cancer risk prediction from symptom survey data: neural-network enhancements and hybrid stacking

Code and results for the paper *Neural Network Enhancements and Hybrid Stacking for Lung Cancer Risk Prediction from Symptom Survey Data: A Rigorous Evaluation under Class Imbalance and Duplicate Records*.

## Data
Download `survey_lung_cancer.csv` from Kaggle: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer
(309 records, 15 predictors; 33 exact duplicates are removed by the code, leaving 276 records.)
Place the file in the same folder as the scripts.

## Scripts
| File | Purpose |
|---|---|
| `lung_cancer_nn_pipeline.py` | Main experiment: 15 models, repeated stratified 5-fold CV (3 repeats, seed 42), metrics, Wilcoxon tests, figures |
| `extra_analysis.py` | Confusion matrices (averaged over repeats) and fold-wise permutation importance |
| `hybrid_improvement.py` | Pre-specified redesigned hybrids and threshold tuning on fresh splits (seed 2026) |

## Run
```
pip install -r requirements.txt
python lung_cancer_nn_pipeline.py survey_lung_cancer.csv results
python extra_analysis.py survey_lung_cancer.csv lung_cancer_nn_pipeline.py results_extra
python hybrid_improvement.py
```
Set the environment variable `SKIP_TUNED=1` to skip the slow nested-CV tuned MLP in the main pipeline.
Tested with Python 3.11 and scikit-learn 1.8.0.

## Results files
`cv_fold_metrics.csv`, `summary_metrics.csv`, `wilcoxon_vs_baseline.csv`, `feature_importance.csv` and `fig_*.png` come from the main pipeline; `hybrid_fold_metrics.csv` comes from `hybrid_improvement.py`.
