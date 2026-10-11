"""Pre-specified hybrid-improvement experiment (new CV seed 2026).
Candidates were fixed BEFORE running; all are reported.
Run (all three files in the same folder):  python hybrid_improvement.py
Or with explicit paths: python hybrid_improvement.py survey_lung_cancer.csv lung_cancer_nn_pipeline.py out_dir"""
import sys, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "survey_lung_cancer.csv")
PIPE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "lung_cancer_nn_pipeline.py")
OUT  = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "results_hybrid")
os.makedirs(OUT, exist_ok=True)
for f in (CSV, PIPE):
    if not os.path.exists(f):
        sys.exit(f"File not found: {f}\nPut survey_lung_cancer.csv and lung_cancer_nn_pipeline.py in the same folder as this script.")
sys.argv = ['x', CSV, OUT]
exec(open(PIPE, encoding="utf-8").read().split("# ------------------------------------------------------------------ main")[0])
from sklearn.model_selection import cross_val_predict
NEW_SEED = 2026

class ThresholdTuned(ClassifierMixin, BaseEstimator):
    """Chooses the threshold maximising balanced accuracy on inner out-of-fold predictions;
    predict_proba is re-centred so that 0.5 equals the chosen threshold (AUC unchanged)."""
    def __init__(self, estimator=None, cv=5, random_state=NEW_SEED):
        self.estimator, self.cv, self.random_state = estimator, cv, random_state
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(clone(self.estimator), X, y, cv=skf, method="predict_proba")[:, 1]
        grid = np.linspace(.05, .95, 91)
        sc = np.array([balanced_accuracy_score(y, (oof >= t).astype(int)) for t in grid])
        self.thr_ = float(grid[sc >= sc.max() - 1e-12].mean())
        self.estimator_ = clone(self.estimator).fit(X, y); self.classes_ = self.estimator_.classes_
        return self
    def predict_proba(self, X):
        p = np.clip(self.estimator_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)) - np.log(self.thr_ / (1 - self.thr_))
        q = 1 / (1 + np.exp(-z)); return np.column_stack([1 - q, q])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= .5).astype(int)

lr  = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
rf  = lambda: RandomForestClassifier(100, class_weight="balanced", random_state=SEED)
gb  = lambda: GradientBoostingClassifier(random_state=SEED)
nn  = lambda: Pipeline([("sc", StandardScaler()), ("m", OverSampled(mlp(hidden_layer_sizes=(16,), alpha=0.3, max_iter=2000)))])
metaLR = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
stackC = lambda: Pipeline([("sc", StandardScaler()), ("clf", StackingClassifier(
    [("lr", lr()), ("rf", rf()), ("gb", gb()), ("nn", nn())], final_estimator=metaLR(),
    stack_method="predict_proba", cv=5, passthrough=False))])
stackD = lambda: Pipeline([("sc", StandardScaler()), ("clf", StackingClassifier(
    [("lr", lr()), ("nn", nn())], final_estimator=metaLR(), stack_method="predict_proba", cv=5))])
bag = lambda: Pipeline([("sc", StandardScaler()), ("clf", OverSampled(BaggingClassifier(
    mlp(hidden_layer_sizes=(16,), alpha=0.3, max_iter=2000), n_estimators=15, max_samples=0.9, bootstrap=True, random_state=SEED)))])

models = {  # cheap first so partial results arrive early
  "LogReg": pipe(lr()),
  "LogReg + tuned threshold": ThresholdTuned(pipe(lr())),
  "Hybrid-D: Stack(LR,NN)->LR": stackD(),
  "Bagged MLP + oversampling": bag(),
  "Hybrid-C: Stack(LR,RF,GB,NN)->LR": stackC(),
  "Bagged MLP + tuned threshold": ThresholdTuned(bag()),
  "Hybrid-C + tuned threshold": ThresholdTuned(stackC()),
}
X, y, _ = load_data(CSV); X = X.values.astype(float)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=NEW_SEED)
splits = list(cv.split(X, y)); rows = []
for name, mdl in models.items():
    t0 = time.time()
    for k, (tr, te) in enumerate(splits):
        m = clone(mdl).fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
        rows.append({"model": name, "fold": k, **metrics(y[te], p)})
    pd.DataFrame(rows).to_csv(f"{OUT}/hybrid_fold_metrics.csv", index=False)
    d = pd.DataFrame([r for r in rows if r["model"] == name]).mean(numeric_only=True)
    print(f"{name:36s} BalAcc={d.BalancedAcc:.3f} MCC={d.MCC:.3f} AUC={d.ROC_AUC:.3f} Spec={d.Specificity:.3f} Rec={d.Recall:.3f} F1={d.F1:.3f}  [{time.time()-t0:.0f}s]", flush=True)
print("done")
