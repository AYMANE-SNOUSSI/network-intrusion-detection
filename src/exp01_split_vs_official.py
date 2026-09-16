"""Experiment 1: the same model, evaluated two ways.

A random split of the training file measures how well the model recognises attacks it has
already seen. The official test file contains 17 attack types absent from training, which is
what deployment looks like. The gap between the two numbers is the result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import CATEGORICAL, FEATURES, NUMERIC, load

SEED = 42
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def make_model() -> Pipeline:
    # Encoder fitted on training data only; categories unseen at test time become all-zeros.
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def metrics(y_true, y_pred, category) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "accuracy": float((tp + tn) / len(y_true)),
        "attack_precision": float(precision_score(y_true, y_pred)),
        "attack_recall": float(recall_score(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)),
        "recall_per_category": {},
    }
    for cat in ["DoS", "Probe", "R2L", "U2R"]:
        mask = np.asarray(category == cat)
        if mask.any():
            out["recall_per_category"][cat] = {
                "recall": float(np.asarray(y_pred)[mask].mean()), "n": int(mask.sum())}
    return out


def main() -> None:
    train, test = load("KDDTrain+.txt"), load("KDDTest+.txt")
    novel = set(test["label"]) - set(train["label"])
    results = {}

    # A. random split inside the training file
    tr, va = train_test_split(train, test_size=0.2, random_state=SEED, stratify=train["label"])
    model = make_model().fit(tr[FEATURES], tr["is_attack"])
    results["A_random_split"] = metrics(va["is_attack"], model.predict(va[FEATURES]), va["category"])

    # B. full training file -> official test file
    model = make_model().fit(train[FEATURES], train["is_attack"])
    pred = model.predict(test[FEATURES])
    results["B_official_test"] = metrics(test["is_attack"], pred, test["category"])
    is_novel = test["label"].isin(novel).to_numpy()
    known_attack = (test["is_attack"].to_numpy() == 1) & ~is_novel
    results["B_official_test"]["recall_known_attack_types"] = float(pred[known_attack].mean())
    results["B_official_test"]["recall_novel_attack_types"] = float(pred[is_novel].mean())

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "exp01_split_vs_official.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
