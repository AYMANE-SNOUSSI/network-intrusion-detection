"""Network intrusion detection API.

    python -m uvicorn api:app --reload        # then open http://127.0.0.1:8000/docs

The model is not trained here: it is loaded from models/model.joblib, produced by train.py.
Training and serving are separate on purpose, so a request never waits for a fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from data import FEATURES
from explain import Explainer

MODELS = Path(__file__).resolve().parents[1] / "models"

# One real attack row from KDDTest+ (a neptune SYN flood), so /docs has something to send.
EXAMPLE = {
    "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "REJ",
    "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
    "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0, "root_shell": 0,
    "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0,
    "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 229, "srv_count": 10, "serror_rate": 0.0, "srv_serror_rate": 0.0,
    "rerror_rate": 1.0, "srv_rerror_rate": 1.0, "same_srv_rate": 0.04, "diff_srv_rate": 0.06,
    "srv_diff_host_rate": 0.0, "dst_host_count": 255, "dst_host_srv_count": 10,
    "dst_host_same_srv_rate": 0.04, "dst_host_diff_srv_rate": 0.06,
    "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 1.0, "dst_host_srv_rerror_rate": 1.0,
}


class Connection(BaseModel):
    """One network connection, described by the 41 NSL-KDD fields.

    Every field is required. A request that omits one, or sends text where a number is
    expected, is rejected with an error naming the field. The internship version replaced
    missing fields with zeros and always produced a prediction, including on files that had
    nothing to do with this data.
    """
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    # connection basics
    duration: int = Field(ge=0, description="seconds the connection lasted")
    protocol_type: str = Field(description="tcp, udp or icmp")
    service: str = Field(description="destination service, e.g. http, private, telnet")
    flag: str = Field(description="how the connection ended, e.g. SF, S0, REJ")
    src_bytes: int = Field(ge=0)
    dst_bytes: int = Field(ge=0)
    land: int = Field(ge=0, le=1)
    wrong_fragment: int = Field(ge=0)
    urgent: int = Field(ge=0)

    # content of the session
    hot: int = Field(ge=0)
    num_failed_logins: int = Field(ge=0)
    logged_in: int = Field(ge=0, le=1)
    num_compromised: int = Field(ge=0)
    root_shell: int = Field(ge=0, le=1)
    su_attempted: int = Field(ge=0, le=2)
    num_root: int = Field(ge=0)
    num_file_creations: int = Field(ge=0)
    num_shells: int = Field(ge=0)
    num_access_files: int = Field(ge=0)
    num_outbound_cmds: int = Field(ge=0)
    is_host_login: int = Field(ge=0, le=1)
    is_guest_login: int = Field(ge=0, le=1)

    # traffic over the last two seconds
    count: int = Field(ge=0)
    srv_count: int = Field(ge=0)
    serror_rate: float = Field(ge=0, le=1)
    srv_serror_rate: float = Field(ge=0, le=1)
    rerror_rate: float = Field(ge=0, le=1)
    srv_rerror_rate: float = Field(ge=0, le=1)
    same_srv_rate: float = Field(ge=0, le=1)
    diff_srv_rate: float = Field(ge=0, le=1)
    srv_diff_host_rate: float = Field(ge=0, le=1)

    # traffic towards the destination host, over the last 100 connections
    dst_host_count: int = Field(ge=0)
    dst_host_srv_count: int = Field(ge=0)
    dst_host_same_srv_rate: float = Field(ge=0, le=1)
    dst_host_diff_srv_rate: float = Field(ge=0, le=1)
    dst_host_same_src_port_rate: float = Field(ge=0, le=1)
    dst_host_srv_diff_host_rate: float = Field(ge=0, le=1)
    dst_host_serror_rate: float = Field(ge=0, le=1)
    dst_host_srv_serror_rate: float = Field(ge=0, le=1)
    dst_host_rerror_rate: float = Field(ge=0, le=1)
    dst_host_srv_rerror_rate: float = Field(ge=0, le=1)


# The fields above and the columns the model was trained on must be the same, in the same order.
# Checked once at import: a mismatch stops the service instead of predicting on shifted columns.
if list(Connection.model_fields) != FEATURES:
    raise RuntimeError("Connection fields do not match the model's features: "
                       f"{set(FEATURES) ^ set(Connection.model_fields)}")


class Prediction(BaseModel):
    prediction: str = Field(description='"attack" or "normal"')
    attack_probability: float = Field(description="0 to 1, how sure the model is it is an attack")
    threshold: float = Field(description="probability above which an alert is raised")
    model_trained_on: str


class Factor(BaseModel):
    feature: str = Field(description="which of the 41 fields")
    value: int | float | str = Field(description="the value that was sent for it")
    contribution: float = Field(description="how much it moved the decision; 0 means no effect")
    direction: str = Field(description='"towards attack" or "towards normal"')


class Explanation(Prediction):
    top_factors: list[Factor] = Field(description="the fields that weighed most, strongest first")


app = FastAPI(
    title="Network intrusion detection",
    description="Predicts whether a network connection is an attack, from the 41 NSL-KDD fields.",
    version="1.0.0",
)

try:
    MODEL = joblib.load(MODELS / "model.joblib")
    CARD = json.loads((MODELS / "model_card.json").read_text())
except FileNotFoundError as error:                     # started before train.py was run
    raise SystemExit(f"{error}\nRun 'python src/train.py' first to create models/model.joblib.")

# Built once at startup (about 0.04 s) rather than per request; explaining one connection
# then costs about 0.02 s, which is small enough to answer inside a request.
EXPLAINER = Explainer(MODEL)


def score(connection: Connection) -> tuple[pd.DataFrame, float]:
    """Shared by /predict and /explain: the input as one row, and its attack probability."""
    row = pd.DataFrame([connection.model_dump()])[FEATURES]
    try:
        return row, float(MODEL.predict_proba(row)[0, 1])
    except Exception as error:                         # a model that cannot score this input
        raise HTTPException(status_code=500, detail=f"prediction failed: {error}") from error


def verdict(probability: float) -> dict:
    threshold = CARD["threshold"]
    return {
        "prediction": "attack" if probability >= threshold else "normal",
        "attack_probability": round(probability, 4),
        "threshold": threshold,
        "model_trained_on": CARD["trained_on"],
    }


@app.get("/health")
def health() -> dict:
    """Is the service alive and does it hold a model? Called by monitoring, not by a human."""
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/model")
def model_card() -> dict:
    """What this model is, how it was trained, and what it scored. Its identity card."""
    return CARD


@app.post("/predict", response_model=Prediction)
def predict(connection: Connection) -> Prediction:
    """Score one connection. Validation happens before this function runs."""
    _, probability = score(connection)
    return Prediction(**verdict(probability))


@app.post("/explain", response_model=Explanation)
def explain(connection: Connection, top_k: int = 5) -> Explanation:
    """The same answer, plus which fields drove it.

    Contributions come from SHAP and are in log-odds: 0 is neutral, a bigger number is a
    stronger push, and the sign says which way. They are what the model used, not the cause
    of the attack. An analyst reads them to decide whether the alert is worth opening.
    """
    row, probability = score(connection)
    factors = EXPLAINER.explain_one(row, top_k=top_k)["top_factors"]
    return Explanation(**verdict(probability), top_factors=factors)