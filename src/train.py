"""Train the model once and save it to a file, so the API never retrains at startup.

    python src/train.py

Writes models/model.joblib (the fitted pipeline) and models/model_card.json (what it is, how it
was trained, what it scored). The API loads the file and does not know how it was produced.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from data import FEATURES, load
from exp02_model_comparison import build

SEED = 0
MODEL_NAME = "hist_gradient_boosting"
THRESHOLD = 0.5
MODELS = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    train, test = load("KDDTrain+.txt"), load("KDDTest+.txt")
    model = build(MODEL_NAME, SEED).fit(train[FEATURES], train["is_attack"])

    predicted = model.predict_proba(test[FEATURES])[:, 1] >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(test["is_attack"], predicted, labels=[0, 1]).ravel()
    card = {
        "model": MODEL_NAME,
        "trained_on": "NSL-KDD KDDTrain+",
        "trained_date": date.today().isoformat(),
        "seed": SEED,
        "threshold": THRESHOLD,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "evaluated_on": "NSL-KDD KDDTest+ (17 attack types absent from training)",
        "accuracy": round(float(accuracy_score(test["is_attack"], predicted)), 4),
        "attack_precision": round(float(precision_score(test["is_attack"], predicted)), 4),
        "attack_recall": round(float(recall_score(test["is_attack"], predicted)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4),
        "known_limitation": ("trained on public 1998-era traffic; the recall above is what this "
                             "model achieves on attack types it has never seen"),
    }

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "model.joblib")
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2))
    size_mb = (MODELS / "model.joblib").stat().st_size / 1e6
    print(json.dumps({k: v for k, v in card.items() if k != "features"}, indent=2))
    print(f"\nsaved models/model.joblib ({size_mb:.1f} MB) and models/model_card.json")


if __name__ == "__main__":
    main()
