"""Per-prediction explanations with SHAP, expressed in the 41 original features.

The model sees one-hot columns (service_http, service_telnet, ...). An analyst thinks in
original fields (service). SHAP values are additive, so the contributions of the one-hot
columns of a field are summed back into that field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from data import CATEGORICAL, FEATURES


class Explainer:
    def __init__(self, model: Pipeline):
        self.pre = model.named_steps["pre"]
        self.clf = model.named_steps["clf"]
        self.tree = shap.TreeExplainer(self.clf)
        transformed = [n.split("__", 1)[1] for n in self.pre.get_feature_names_out()]
        # map every transformed column back to its original field
        self.field_of = [next((c for c in CATEGORICAL if t.startswith(c + "_")), t) for t in transformed]
        base = np.asarray(self.tree.expected_value).ravel()
        self.base_value = float(base[-1])

    def contributions(self, rows: pd.DataFrame) -> pd.DataFrame:
        """SHAP values in log-odds, one column per original field."""
        x = self.pre.transform(rows[FEATURES])
        values = np.asarray(self.tree.shap_values(x))
        if values.ndim == 3:                       # some shap versions return one array per class
            values = values[..., -1]
        # additivity check: base + sum(contributions) must equal the model's raw output
        raw = self.clf.decision_function(x)
        if not np.allclose(self.base_value + values.sum(axis=1), raw, atol=1e-4):
            raise RuntimeError("SHAP values do not add up to the model output")
        df = pd.DataFrame(values, columns=self.field_of, index=rows.index)
        return df.T.groupby(level=0, sort=False).sum().T[FEATURES]

    def explain_one(self, row: pd.DataFrame, top_k: int = 5) -> dict:
        """What an analyst receives with an alert: the fields that pushed the decision most."""
        contrib = self.contributions(row).iloc[0]
        order = contrib.abs().sort_values(ascending=False).index[:top_k]
        prob = float(self.clf.predict_proba(self.pre.transform(row[FEATURES]))[0, 1])
        return {
            "attack_probability": prob,
            "top_factors": [
                {"feature": f, "value": row.iloc[0][f].item() if hasattr(row.iloc[0][f], "item") else row.iloc[0][f],
                 "contribution": round(float(contrib[f]), 3),
                 "direction": "towards attack" if contrib[f] > 0 else "towards normal"}
                for f in order
            ],
        }
