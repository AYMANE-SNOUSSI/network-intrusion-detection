"""Experiment 2: does the choice of model matter, once the evaluation is honest?

Every model is trained on KDDTrain+ and evaluated on KDDTest+ and on KDDTest-21 (the subset
of test records that most classifiers in the original NSL-KDD study got wrong).

Includes the design of the original internship version (SelectKBest, k=10, then RandomForest)
so the effect of that feature selection is measured rather than assumed.
Tree models are run over several seeds: a difference smaller than the seed spread is not a
difference.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data import CATEGORICAL, FEATURES, NUMERIC, load
from exp01_split_vs_official import metrics

SEEDS = [0, 1, 2]
REPORTS = Path(__file__).resolve().parents[1] / "reports"
warnings.filterwarnings("ignore", category=UserWarning)      # constant one-hot columns in f_classif
warnings.filterwarnings("ignore", category=RuntimeWarning)


def preprocessor(scale: bool) -> ColumnTransformer:
    # Byte counts span 0 to 1e9: log1p before scaling, or the linear model sees only outliers.
    num = make_pipeline(FunctionTransformer(np.log1p), StandardScaler()) if scale else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", num, NUMERIC),
    ])


def build(name: str, seed: int) -> Pipeline:
    if name == "majority_class":
        return Pipeline([("pre", preprocessor(False)), ("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "logistic_regression":
        return Pipeline([("pre", preprocessor(True)),
                         ("clf", LogisticRegression(max_iter=2000, random_state=seed))])
    if name == "random_forest_41":
        return Pipeline([("pre", preprocessor(False)),
                         ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))])
    if name == "random_forest_k10_original":
        return Pipeline([("pre", preprocessor(False)), ("sel", SelectKBest(f_classif, k=10)),
                         ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))])
    if name == "hist_gradient_boosting":
        return Pipeline([("pre", preprocessor(False)),
                         ("clf", HistGradientBoostingClassifier(random_state=seed))])
    raise ValueError(name)


MODELS = {  # name -> is the result seed-dependent?
    "majority_class": False,
    "logistic_regression": False,
    "random_forest_41": True,
    "random_forest_k10_original": True,
    "hist_gradient_boosting": True,
}


def summarise(runs: list[dict]) -> dict:
    """Mean and std over seeds for the headline numbers."""
    keys = ["accuracy", "attack_recall", "false_positive_rate", "recall_R2L", "recall_novel_attack_types"]
    return {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs]))} for k in keys}


def main() -> None:
    train = load("KDDTrain+.txt")
    tests = {"KDDTest+": load("KDDTest+.txt"), "KDDTest-21": load("KDDTest-21.txt")}
    known = set(train["label"])
    results: dict = {}

    for name, seeded in MODELS.items():
        results[name] = {}
        runs = {t: [] for t in tests}
        for seed in (SEEDS if seeded else [SEEDS[0]]):
            model = build(name, seed)
            t0 = time.perf_counter()
            model.fit(train[FEATURES], train["is_attack"])
            fit_s = time.perf_counter() - t0
            for tname, test in tests.items():
                t0 = time.perf_counter()
                pred = model.predict(test[FEATURES])
                pred_ms = 1000 * (time.perf_counter() - t0) / len(test)
                m = metrics(test["is_attack"], pred, test["category"])
                novel = (~test["label"].isin(known)).to_numpy()
                m["recall_R2L"] = m["recall_per_category"].get("R2L", {}).get("recall", float("nan"))
                m["recall_novel_attack_types"] = float(pred[novel].mean())
                m["fit_seconds"], m["predict_ms_per_row"], m["seed"] = fit_s, pred_ms, seed
                runs[tname].append(m)
            print(f"{name:28s} seed={seed}  "
                  + "  ".join(f"{t}: acc={runs[t][-1]['accuracy']:.4f} "
                              f"rec={runs[t][-1]['attack_recall']:.4f} "
                              f"fpr={runs[t][-1]['false_positive_rate']:.4f}" for t in tests)
                  + f"  fit={fit_s:.1f}s", flush=True)
        if name == "random_forest_k10_original":
            sel = model.named_steps["sel"].get_support()
            names = model.named_steps["pre"].get_feature_names_out()
            results[name]["selected_features"] = [n.split("__", 1)[1] for n in names[sel]]
        for tname in tests:
            results[name][tname] = {"summary": summarise(runs[tname]), "runs": runs[tname]}

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "exp02_model_comparison.json").write_text(json.dumps(results, indent=2))

    print("\nKDDTest+ (mean ± std over seeds)")
    print(f"{'model':28s} {'accuracy':>16s} {'attack recall':>16s} {'FPR':>16s} {'R2L recall':>16s} {'novel recall':>16s}")
    for name in MODELS:
        s = results[name]["KDDTest+"]["summary"]
        print(f"{name:28s} " + " ".join(f"{s[k]['mean']:>9.4f} ±{s[k]['std']:.4f}" for k in
              ["accuracy", "attack_recall", "false_positive_rate", "recall_R2L", "recall_novel_attack_types"]))


if __name__ == "__main__":
    main()
