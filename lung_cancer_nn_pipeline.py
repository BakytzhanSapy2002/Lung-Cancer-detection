#!/usr/bin/env python3
"""
Lung cancer survey dataset: improving neural-network classification.
Reproducible pipeline for the paper's "Results" section.

Usage:  python lung_cancer_nn_pipeline.py [path/to/survey_lung_cancer.csv] [out_dir]
Needs:  numpy pandas scikit-learn matplotlib scipy   (no GPU / torch required)

What it does
  1. Cleans data (removes duplicate rows -> avoids train/test leakage), encodes features.
  2. Compares 8 models with repeated stratified 5-fold CV (3 repeats):
       LR, RF, GB, baseline MLP, tuned MLP (nested CV), bagged MLP,
       oversampled bagged MLP, soft-voting ensemble (bagged MLP + LR + GB).
  3. Metrics: Accuracy, Balanced Acc, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC.
  4. Statistical test (Wilcoxon) of every model vs. baseline MLP.
  5. Figures: ROC curves, metric bars, confusion matrix, permutation importance.
All oversampling / scaling happens INSIDE each training fold (no leakage).
"""
import sys, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (StackingClassifier, BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "survey_lung_cancer.csv"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    info = {"n_raw": n_raw, "n_after_dedup": len(df), "duplicates_removed": n_raw - len(df)}
    df["GENDER"] = (df["GENDER"].str.strip() == "M").astype(int)
    y = (df["LUNG_CANCER"].str.strip() == "YES").astype(int).values
    X = df.drop(columns="LUNG_CANCER")
    bin_cols = [c for c in X.columns if c not in ("AGE", "GENDER")]
    X[bin_cols] = X[bin_cols] - 1                       # 1/2 -> 0/1
    info["class_counts"] = {"YES": int(y.sum()), "NO": int((1 - y).sum())}
    return X, y, info


# ------------------- knowledge-guided composite risk scores (hybrid, part 1)
# column order: 0 GENDER 1 AGE 2 SMOKING 3 YELLOW_FINGERS 4 ANXIETY 5 PEER_PRESSURE
# 6 CHRONIC_DISEASE 7 FATIGUE 8 ALLERGY 9 WHEEZING 10 ALCOHOL 11 COUGHING
# 12 SHORTNESS_OF_BREATH 13 SWALLOWING_DIFFICULTY 14 CHEST_PAIN
RESP_IDX = [7, 9, 11, 12, 13, 14]     # respiratory / systemic symptom burden
LIFE_IDX = [2, 3, 5, 10]              # smoking-related & lifestyle exposure


class AddRiskScores(BaseEstimator):
    """x -> [x, S_resp(x), S_life(x), S_resp*S_life]  (S = sum of binary indicators)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        sr, sl = X[:, RESP_IDX].sum(1), X[:, LIFE_IDX].sum(1)
        return np.column_stack([X, sr, sl, sr * sl])

    def fit_transform(self, X, y=None):
        return self.transform(X)


# ------------------------------------------------ in-fold oversampling
class OverSampled(ClassifierMixin, BaseEstimator):
    """Random-oversamples the minority class in fit() only (never at predict)."""
    def __init__(self, estimator=None, random_state=SEED):
        self.estimator, self.random_state = estimator, random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        cls, cnt = np.unique(y, return_counts=True)
        minority, majority_n = cls[cnt.argmin()], cnt.max()
        idx_min = np.where(y == minority)[0]
        extra = rng.choice(idx_min, majority_n - len(idx_min), replace=True)
        idx = np.concatenate([np.arange(len(y)), extra])
        self.estimator_ = clone(self.estimator).fit(X[idx], y[idx])
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


# ---------------------------------------------------------------- models
def mlp(**kw):
    p = dict(hidden_layer_sizes=(16,), alpha=1e-4, max_iter=1000, random_state=SEED)
    p.update(kw)
    return MLPClassifier(**p)


def pipe(est):
    return Pipeline([("sc", StandardScaler()), ("clf", est)])


def build_models():
    # NOTE: early_stopping=True collapsed to "all positive" on this imbalanced data
    # (documented as an ablation: model "Bagged MLP (early-stop, no oversampling)").
    best_cfg = dict(hidden_layer_sizes=(16,), alpha=0.3, max_iter=2000)
    bad_cfg = dict(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True,
                   n_iter_no_change=20, learning_rate="adaptive", validation_fraction=0.15)
    bag_bad = BaggingClassifier(mlp(**bad_cfg), n_estimators=15, max_samples=0.9,
                                bootstrap=True, random_state=SEED)
    bag_mlp = BaggingClassifier(mlp(**best_cfg), n_estimators=15, max_samples=0.9,
                                bootstrap=True, random_state=SEED)
    grid = {"clf__estimator__hidden_layer_sizes": [(8,), (16,), (32, 16)],
            "clf__estimator__alpha": [0.03, 0.3, 1.0]}
    tuned = GridSearchCV(pipe(OverSampled(mlp(max_iter=2000))),
                         grid, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True,
                                                                     random_state=SEED))
    base_lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
    base_rf = lambda: RandomForestClassifier(200, class_weight="balanced", random_state=SEED)
    base_gb = lambda: GradientBoostingClassifier(random_state=SEED)
    final_nn = lambda: OverSampled(mlp(hidden_layer_sizes=(16,), alpha=0.3, max_iter=2000))

    def stack(passthrough=True):
        return StackingClassifier(
            [("lr", base_lr()), ("rf", base_rf()), ("gb", base_gb())],
            final_estimator=final_nn(), stack_method="predict_proba", cv=5,
            passthrough=passthrough, n_jobs=1)

    models = {
        "LogReg": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "SVM (RBF)": pipe(SVC(probability=True, class_weight="balanced", random_state=SEED)),
        "kNN": pipe(KNeighborsClassifier(15, weights="distance")),
        "GaussianNB": pipe(GaussianNB()),
        "RandomForest": pipe(RandomForestClassifier(300, class_weight="balanced",
                                                    random_state=SEED)),
        "GradBoost": pipe(GradientBoostingClassifier(random_state=SEED)),
        "MLP baseline": pipe(mlp()),
        "MLP tuned (nested CV)": tuned,
        "Bagged MLP (early-stop, no oversampling)": pipe(bag_bad),
        "MLP + oversampling": pipe(OverSampled(mlp(**best_cfg))),
        "Bagged MLP": pipe(bag_mlp),
        "Bagged MLP + oversampling": pipe(OverSampled(clone(bag_mlp))),
        "Soft-voting (BagMLP+LR+GB)": pipe(OverSampled(VotingClassifier(
            [("bmlp", clone(bag_mlp)),
             ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
             ("gb", GradientBoostingClassifier(random_state=SEED))], voting="soft"))),
        # ---- proposed hybrids
        "Hybrid-A: Stack(LR,RF,GB)->MLP": pipe(stack()),
        "Hybrid-B: RiskScores + Stack->MLP (proposed)": Pipeline(
            [("rs", AddRiskScores()), ("sc", StandardScaler()), ("clf", stack())]),
    }
    if os.environ.get("SKIP_TUNED"):
        models.pop("MLP tuned (nested CV)")
    return models


def metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(Accuracy=accuracy_score(y, pred), BalancedAcc=balanced_accuracy_score(y, pred),
                Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred), F1=f1_score(y, pred),
                MCC=matthews_corrcoef(y, pred), ROC_AUC=roc_auc_score(y, proba),
                PR_AUC=average_precision_score(y, proba),
                Specificity=recall_score(y, pred, pos_label=0))


# ------------------------------------------------------------------ main
def main():
    X, y, info = load_data(CSV)
    print("DATA:", json.dumps(info))
    X = X.values.astype(float); feat = list(load_data(CSV)[0].columns)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    models = build_models()

    rows, oof = [], {n: np.zeros((3, len(y))) for n in models}
    for name, model in models.items():
        print(f"-> {name}", flush=True)
        for k, (tr, te) in enumerate(cv.split(X, y)):
            m = clone(model).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            oof[name][k // 5, te] = p
            rows.append({"model": name, "fold": k, **metrics(y[te], p)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cv_fold_metrics.csv", index=False)

    summ = res.groupby("model").agg(["mean", "std"]).drop(columns="fold")
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.sort_values("ROC_AUC_mean", ascending=False)
    summ.round(4).to_csv(OUT / "summary_metrics.csv")
    print("\n", summ[[c for c in summ.columns if c.endswith("_mean")]].round(3).to_string())

    # Wilcoxon vs baseline MLP (paired over 15 folds)
    base = res[res.model == "MLP baseline"].sort_values("fold")
    st = []
    for name in models:
        if name == "MLP baseline":
            continue
        cur = res[res.model == name].sort_values("fold")
        for met in ("ROC_AUC", "F1", "BalancedAcc"):
            d = cur[met].values - base[met].values
            try:
                pval = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pval = 1.0
            st.append({"model": name, "metric": met, "mean_diff": d.mean(), "p_value": pval})
    pd.DataFrame(st).round(4).to_csv(OUT / "wilcoxon_vs_baseline.csv", index=False)

    # ---- figures
    best = summ.index[0]
    plt.figure(figsize=(7, 6))
    for name in models:
        p = oof[name].mean(axis=0)               # average OOF prob over repeats
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves (out-of-fold)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(OUT / "fig_roc.png", dpi=200); plt.close()

    ms = ["BalancedAcc", "F1", "MCC", "ROC_AUC"]
    ax = summ[[f"{m}_mean" for m in ms]].rename(columns=lambda c: c[:-5]).plot.bar(
        figsize=(10, 5), yerr=None)
    ax.set_title("Cross-validated metrics (mean of 15 folds)"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(OUT / "fig_metrics.png", dpi=200); plt.close()

    pbest = oof[best].mean(axis=0)
    cm = confusion_matrix(y, (pbest >= .5).astype(int))
    plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["No", "Yes"]); plt.yticks([0, 1], ["No", "Yes"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix\n{best}", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "fig_confusion.png", dpi=200); plt.close()

    # permutation importance of the best model (hold-out 25 %)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    m = clone(models[best]).fit(Xtr, ytr)
    pi = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=30, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=feat).sort_values()
    plt.figure(figsize=(6, 5)); imp.plot.barh(xerr=pi.importances_std[imp.index.map(feat.index)])
    plt.title(f"Permutation importance (ΔAUC), {best}", fontsize=9); plt.tight_layout()
    plt.savefig(OUT / "fig_importance.png", dpi=200); plt.close()
    imp.sort_values(ascending=False).round(4).to_csv(OUT / "feature_importance.csv")
    print("\nBest model:", best, "| files saved in", OUT.resolve())


if __name__ == "__main__":
    main()
