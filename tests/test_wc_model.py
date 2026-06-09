"""Tests unitaires du modèle sur données synthétiques (rapides, déterministes)."""
import numpy as np
import pandas as pd
import wc_model as wm


def _synthetic():
    """Ligue jouet : 4 équipes de forces décroissantes, ~800 matchs neutres."""
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 2.0, "B": 1.2, "C": 0.9, "D": 0.5}
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(800):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({
            "date": base + pd.Timedelta(days=i),
            "home_team": h, "away_team": a,
            "home_score": int(rng.poisson(strength[h])),
            "away_score": int(rng.poisson(strength[a])),
            "tournament": "Test", "neutral": True,
        })
    return pd.DataFrame(rows)


DATE_REF = pd.Timestamp("2023-01-01")


def test_predict_probabilities_sum_to_one():
    m = wm.fit_model(_synthetic(), date_ref=DATE_REF, estimate_rho=False, lambda_ridge=0.1)
    r = m.predict("A", "D")
    assert abs(r["p1"] + r["pN"] + r["p2"] - 1.0) < 1e-6


def test_strong_beats_weak():
    m = wm.fit_model(_synthetic(), date_ref=DATE_REF, estimate_rho=False, lambda_ridge=0.1)
    r = m.predict("A", "D")
    assert r["p1"] > r["p2"]


def test_rho_zero_when_not_estimated():
    m = wm.fit_model(_synthetic(), date_ref=DATE_REF, estimate_rho=False, lambda_ridge=0.1)
    assert m.rho == 0.0


def test_rho_estimated_finite_in_range():
    # L'estimation 1-D doit renvoyer un rho fini, dans les bornes de recherche.
    m = wm.fit_model(_synthetic(), date_ref=DATE_REF, estimate_rho=True, lambda_ridge=0.1)
    assert np.isfinite(m.rho)
    assert -0.2 <= m.rho <= 0.2


def test_dixon_coles_correction_normalized_and_shifts_draw():
    # Test déterministe de l'application de tau : on impose un rho connu et on
    # vérifie que la matrice reste normalisée et que la proba de nul change.
    m = wm.fit_model(_synthetic(), date_ref=DATE_REF, estimate_rho=False, lambda_ridge=0.1)
    r_indep = m.predict("B", "C")
    m.rho = -0.1  # correction Dixon-Coles connue
    r_dc = m.predict("B", "C")
    assert abs(r_dc["p1"] + r_dc["pN"] + r_dc["p2"] - 1.0) < 1e-6   # renormalisée
    assert r_dc["pN"] != r_indep["pN"]                             # le nul bouge
