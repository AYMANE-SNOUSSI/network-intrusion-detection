"""Experiment 3: choosing the alert threshold, and whether that choice survives deployment.

A classifier outputs a probability; the alert fires above a threshold. Lower threshold: more
attacks caught, more false alerts. The threshold must be chosen on data the model has not
been evaluated on. Here it is chosen on a validation split of KDDTrain+ for a target false
positive rate, then applied unchanged to KDDTest+, which stands in for deployment.
Daily volumes use a realistic base rate: 10 000 connections, 50 of them attacks.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from data import FEATURES, load
from exp02_model_comparison import build

SEED = 0
TARGET_FPR = [0.001, 0.01, 0.05]
DAILY_CONNECTIONS, DAILY_ATTACKS = 10_000, 50
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def operating_point(y, score, threshold) -> dict:
    y, alert = np.asarray(y), np.asarray(score) >= threshold
    recall = float(alert[y == 1].mean())
    fpr = float(alert[y == 0].mean())
    false_alerts = fpr * (DAILY_CONNECTIONS - DAILY_ATTACKS)
    caught = recall * DAILY_ATTACKS
    return {
        "recall": recall,
        "fpr": fpr,
        "per_day": {
            "attacks_caught": round(caught, 1),
            "attacks_missed": round(DAILY_ATTACKS - caught, 1),
            "false_alerts": round(false_alerts, 1),
            "share_of_alerts_that_are_real": round(caught / (caught + false_alerts), 3)
            if caught + false_alerts else None,
        },
    }


def threshold_for_fpr(y, score, target) -> float:
    """Smallest threshold whose FPR on (y, score) does not exceed target."""
    normal = np.sort(np.asarray(score)[np.asarray(y) == 0])[::-1]
    k = int(np.floor(target * len(normal)))          # normals allowed above threshold
    return float(np.nextafter(normal[k], np.inf)) if k < len(normal) else 0.0


def main() -> None:
    train, test = load("KDDTrain+.txt"), load("KDDTest+.txt")
    tr, va = train_test_split(train, test_size=0.2, random_state=SEED, stratify=train["label"])
    model = build("hist_gradient_boosting", SEED).fit(tr[FEATURES], tr["is_attack"])
    s_va = model.predict_proba(va[FEATURES])[:, 1]
    s_te = model.predict_proba(test[FEATURES])[:, 1]

    results = {"model": "hist_gradient_boosting", "seed": SEED,
               "roc_auc": {"validation": float(roc_auc_score(va["is_attack"], s_va)),
                           "test": float(roc_auc_score(test["is_attack"], s_te))},
               "operating_points": []}
    points = [("default 0.5", 0.5)] + [(f"FPR {t:g} on validation", threshold_for_fpr(va["is_attack"], s_va, t))
                                       for t in TARGET_FPR]
    for name, thr in points:
        results["operating_points"].append({
            "name": name, "threshold": thr,
            "validation": operating_point(va["is_attack"], s_va, thr),
            "test": operating_point(test["is_attack"], s_te, thr),
        })

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "exp03_threshold.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, y, s in [("validation (KDDTrain+ split)", va["is_attack"], s_va), ("test (KDDTest+)", test["is_attack"], s_te)]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=label)
    for p in results["operating_points"]:
        ax.scatter(p["test"]["fpr"], p["test"]["recall"], zorder=3)
        ax.annotate(p["name"], (p["test"]["fpr"], p["test"]["recall"]), fontsize=7,
                    xytext=(4, -10), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("false positive rate (symlog)"); ax.set_ylabel("attack recall")
    ax.set_title("Thresholds chosen on validation, applied to test"); ax.legend(loc="lower right", fontsize=8)
    (REPORTS / "figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(REPORTS / "figures" / "exp03_roc_thresholds.png", dpi=150)

    print(f"ROC-AUC validation {results['roc_auc']['validation']:.4f}  test {results['roc_auc']['test']:.4f}\n")
    print(f"{'operating point':26s} {'thr':>8s} | {'val FPR':>8s} {'val rec':>8s} | {'test FPR':>8s} {'test rec':>8s} | "
          f"{'caught/day':>10s} {'missed/day':>10s} {'false/day':>10s} {'real share':>10s}")
    for p in results["operating_points"]:
        v, t, d = p["validation"], p["test"], p["test"]["per_day"]
        print(f"{p['name']:26s} {p['threshold']:8.4f} | {v['fpr']:8.4f} {v['recall']:8.4f} | {t['fpr']:8.4f} {t['recall']:8.4f} | "
              f"{d['attacks_caught']:10.1f} {d['attacks_missed']:10.1f} {d['false_alerts']:10.1f} {d['share_of_alerts_that_are_real']!s:>10s}")


if __name__ == "__main__":
    main()
