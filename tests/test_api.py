"""Tests for the API, run without starting a server.

    python -m pytest -v

TestClient calls the functions in api.py directly, so there is no uvicorn, no port and no
network. That is what lets these tests run in continuous integration, where nothing is served.

The two rows below come from KDDTest+, the official test set: one attack the model catches,
one normal connection. They are written out here so the tests never depend on the dataset
being downloaded.
"""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from api import EXAMPLE, app
from data import FEATURES

client = TestClient(app)

# A neptune SYN flood: 229 connections to the same host, every one refused (flag REJ).
ATTACK = EXAMPLE

# An ordinary FTP data transfer of 12983 bytes.
NORMAL = {
    "duration": 2, "protocol_type": "tcp", "service": "ftp_data", "flag": "SF",
    "src_bytes": 12983, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
    "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0, "root_shell": 0,
    "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0,
    "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0,
    "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0, "dst_host_count": 134, "dst_host_srv_count": 86,
    "dst_host_same_srv_rate": 0.61, "dst_host_diff_srv_rate": 0.04,
    "dst_host_same_src_port_rate": 0.61, "dst_host_srv_diff_host_rate": 0.02,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
}


def without(field: str) -> dict:
    """A copy of the attack row with one field removed."""
    body = copy.deepcopy(ATTACK)
    del body[field]
    return body


def replacing(field: str, value) -> dict:
    """A copy of the attack row with one field changed."""
    body = copy.deepcopy(ATTACK)
    body[field] = value
    return body


# --- the service is up -------------------------------------------------------------------

def test_health_repond_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_annonce_ses_limites():
    """The card must say what the model was trained on; a card without it is not honest."""
    card = client.get("/model").json()
    assert card["trained_on"] == "NSL-KDD KDDTrain+"
    assert "known_limitation" in card
    assert 0 < card["attack_recall"] < 1


def test_route_inconnue_donne_404():
    assert client.get("/predictions").status_code == 404


# --- the predictions are the expected ones -----------------------------------------------

def test_predict_signale_une_attaque():
    response = client.post("/predict", json=ATTACK)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "attack"
    assert body["attack_probability"] > 0.99


def test_predict_laisse_passer_une_connexion_normale():
    body = client.post("/predict", json=NORMAL).json()
    assert body["prediction"] == "normal"
    assert body["attack_probability"] < 0.5


def test_le_verdict_suit_le_seuil():
    """prediction is not a separate opinion: it is the probability compared to threshold."""
    for row in (ATTACK, NORMAL):
        body = client.post("/predict", json=row).json()
        expected = "attack" if body["attack_probability"] >= body["threshold"] else "normal"
        assert body["prediction"] == expected


# --- a bad request is refused, it never produces a prediction -----------------------------
# This is the defect of the 2024 version: it replaced what was missing with zeros and
# answered anyway. Each test below must get 422, never 200.

def test_champ_manquant_refuse():
    response = client.post("/predict", json=without("src_bytes"))
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "src_bytes"]


def test_texte_a_la_place_d_un_nombre_refuse():
    response = client.post("/predict", json=replacing("src_bytes", "beaucoup"))
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "src_bytes"]


def test_champ_inconnu_refuse():
    """A field the model was never trained on means the caller is sending the wrong data."""
    response = client.post("/predict", json=replacing("user_id", 42))
    assert response.status_code == 422
    assert "Extra inputs are not permitted" in response.text


def test_taux_hors_intervalle_refuse():
    """same_srv_rate is a proportion: 5.0 is impossible, so it is a mistake upstream."""
    response = client.post("/predict", json=replacing("same_srv_rate", 5.0))
    assert response.status_code == 422


def test_valeur_negative_refuse():
    assert client.post("/predict", json=replacing("duration", -1)).status_code == 422


def test_corps_vide_refuse():
    response = client.post("/predict", json={})
    assert response.status_code == 422
    assert len(response.json()["detail"]) == len(FEATURES)   # all 41 reported at once


# --- the explanation -----------------------------------------------------------------------

def test_explain_rend_le_meme_verdict_que_predict():
    predicted = client.post("/predict", json=ATTACK).json()
    explained = client.post("/explain", json=ATTACK).json()
    assert explained["attack_probability"] == predicted["attack_probability"]
    assert explained["prediction"] == predicted["prediction"]


def test_explain_nomme_les_champs_qui_ont_pese():
    factors = client.post("/explain", json=ATTACK).json()["top_factors"]
    assert len(factors) == 5
    assert all(f["feature"] in FEATURES for f in factors)
    assert all(f["direction"] in ("towards attack", "towards normal") for f in factors)


def test_les_facteurs_sont_tries_du_plus_fort_au_plus_faible():
    factors = client.post("/explain", json=ATTACK).json()["top_factors"]
    poids = [abs(f["contribution"]) for f in factors]
    assert poids == sorted(poids, reverse=True)


@pytest.mark.parametrize("top_k", [1, 3, 10])
def test_top_k_fixe_le_nombre_de_facteurs(top_k):
    factors = client.post(f"/explain?top_k={top_k}", json=ATTACK).json()["top_factors"]
    assert len(factors) == top_k


def test_explain_refuse_aussi_une_entree_invalide():
    assert client.post("/explain", json=without("flag")).status_code == 422


# --- the contract between the API and the model -------------------------------------------

def test_les_41_champs_sont_ceux_du_modele():
    """If someone adds a field to the API without retraining, this test fails immediately."""
    envoyes = list(client.get("/openapi.json").json()
                   ["components"]["schemas"]["Connection"]["properties"])
    assert envoyes == FEATURES
