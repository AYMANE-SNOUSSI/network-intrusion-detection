"""Experiment 4: what the model looks at, and why it misses intrusions.

1. Global view: which fields drive decisions overall.
2. Local view: the explanation attached to a single alert (what the API will return).
3. Diagnosis: R2L attacks (remote intrusions) are almost never caught. SHAP shows what the
   model sees on the missed ones; comparing training and test records shows why.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import FEATURES, load
from exp02_model_comparison import build
from explain import Explainer

SEED, THRESHOLD = 0, 0.5
REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIG = REPORTS / "figures"


def barh(series: pd.Series, title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#c0392b" if v > 0 else "#2e86c1" for v in series.values]
    ax.barh(series.index[::-1], series.values[::-1], color=colors[::-1])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(title); ax.set_xlabel(xlabel)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    train, test = load("KDDTrain+.txt"), load("KDDTest+.txt")
    model = build("hist_gradient_boosting", SEED).fit(train[FEATURES], train["is_attack"])
    test["score"] = model.predict_proba(test[FEATURES])[:, 1]
    test["alert"] = test["score"] >= THRESHOLD
    explainer = Explainer(model)
    results: dict = {"model": "hist_gradient_boosting", "seed": SEED, "threshold": THRESHOLD}
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. global importance
    contrib = explainer.contributions(test)
    importance = contrib.abs().mean().sort_values(ascending=False)
    results["global_importance_top15"] = importance.head(15).round(4).to_dict()
    fig, ax = plt.subplots(figsize=(7, 5))
    top = importance.head(15)
    ax.barh(top.index[::-1], top.values[::-1], color="#555")
    ax.set_title("Global importance on KDDTest+"); ax.set_xlabel("mean |SHAP| (log-odds)")
    fig.tight_layout(); fig.savefig(FIG / "exp04_global_importance.png", dpi=150); plt.close(fig)

    # 2. two local explanations: the most confident true alert, and a missed brute-force attack
    caught = test[(test["is_attack"] == 1) & test["alert"]].sort_values("score").iloc[[-1]]
    missed = test[(test["label"] == "guess_passwd") & ~test["alert"]].iloc[[0]]
    results["example_caught_attack"] = {"label": caught["label"].iloc[0], **explainer.explain_one(caught)}
    results["example_missed_attack"] = {"label": missed["label"].iloc[0], **explainer.explain_one(missed)}

    # 3. R2L diagnosis: detection rate per attack type vs how many examples training had
    r2l = test[test["category"] == "R2L"]
    per_type = (r2l.groupby("label")["alert"].agg(n_test="size", detection_rate="mean")
                .join(train["label"].value_counts().rename("n_train"))
                .fillna({"n_train": 0}).astype({"n_train": int})
                .sort_values("n_test", ascending=False))
    results["r2l_per_attack_type"] = per_type.round(3).to_dict(orient="index")

    # guess_passwd: same name in both files, but is it the same traffic?
    cols = ["service", "flag", "num_failed_logins", "hot"]
    profile = {}
    for name, df in [("train", train), ("test", test)]:
        g = df[df["label"] == "guess_passwd"]
        profile[name] = {
            "n": int(len(g)),
            "service": g["service"].value_counts(normalize=True).head(3).round(3).to_dict(),
            "flag": g["flag"].value_counts(normalize=True).head(3).round(3).to_dict(),
            "num_failed_logins_mean": round(float(g["num_failed_logins"].mean()), 3),
            "hot_mean": round(float(g["hot"].mean()), 3),
        }
    results["guess_passwd_train_vs_test"] = profile

    missed_gp = test[(test["label"] == "guess_passwd") & ~test["alert"]]
    mean_contrib = contrib.loc[missed_gp.index].mean()
    top = mean_contrib.reindex(mean_contrib.abs().sort_values(ascending=False).index[:10])
    results["missed_guess_passwd_mean_contribution_top10"] = top.round(3).to_dict()
    barh(top, f"Missed guess_passwd (n={len(missed_gp)}): what the model sees",
         "mean SHAP (log-odds)   blue = towards normal, red = towards attack",
         FIG / "exp04_missed_guess_passwd.png")

    (REPORTS / "exp04_shap.json").write_text(json.dumps(results, indent=2, default=str))

    print("Top 10 fields overall (mean |SHAP|):")
    for f, v in importance.head(10).items():
        print(f"  {f:28s} {v:.3f}")
    print("\nR2L detection per attack type:")
    print(per_type.head(8).to_string())
    print("\nguess_passwd, train vs test:")
    for k in ["train", "test"]:
        p = profile[k]
        print(f"  {k:5s} n={p['n']:5d} service={p['service']} flag={p['flag']} "
              f"failed_logins={p['num_failed_logins_mean']} hot={p['hot_mean']}")
    print("\nExample explanation, missed attack:")
    print(json.dumps(results["example_missed_attack"], indent=2, default=str))


if __name__ == "__main__":
    main()
