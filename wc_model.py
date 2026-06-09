"""
Modèle de Poisson Maher / Dixon-Coles pour la Coupe du Monde 2026.

Module réutilisable partagé par predict_wc2026.py (pronostics) et validate.py
(validation hors-échantillon), afin que les deux utilisent exactement le même
code d'ajustement.

  log(lambda) = const + attaque[equipe] + defense[adversaire] + domicile * dom

Correction Dixon-Coles (ρ) : voir _estimate_rho et FittedModel.score_matrix.
"""
import os
import urllib.request
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize, minimize_scalar

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

WINDOW_START = "2015-01-01"
HALF_LIFE_DAYS = 730        # poids /2 tous les 2 ans
FRIENDLY_WEIGHT = 0.5       # un amical compte moitié moins
MAX_GOALS = 10
LAMBDA_RIDGE = 1.0          # pénalité L2 sur les forces d'équipes


def load_data(path="results.csv"):
    """Charge le dataset (téléchargement auto si absent)."""
    if not os.path.exists(path):
        print("Telechargement du dataset...")
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path, parse_dates=["date"])


def make_train(df, window_start=WINDOW_START, date_ref=None, min_matches=20, keep_teams=None):
    """Matchs joués dans la fenêtre, en écartant les micro-sélections trop rares
    (records 'parfaits' -> divergence), tout en gardant impérativement keep_teams."""
    train = df[(df["date"] >= window_start)
               & (df["date"] < date_ref)
               & df["home_score"].notna()].copy()
    counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    actives = set(counts[counts >= min_matches].index)
    if keep_teams:
        actives |= set(keep_teams)
    return train[train["home_team"].isin(actives) & train["away_team"].isin(actives)].copy()


def _build_exog(frame, cat, cols=None):
    """Matrice de design : const + domicile + attaque[eq] + defense[adv]."""
    att = pd.get_dummies(frame["equipe"].astype(cat), prefix="att", drop_first=True, dtype=float)
    deff = pd.get_dummies(frame["adv"].astype(cat), prefix="def", drop_first=True, dtype=float)
    X = pd.concat([
        pd.Series(1.0, index=frame.index, name="const"),
        frame["domicile"].astype(float).rename("domicile"),
        att, deff,
    ], axis=1)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


class FittedModel:
    """Modèle ajusté : coefficients + ρ (0 = Poisson indépendant)."""

    def __init__(self, beta, cols, cat, rho=0.0, max_goals=MAX_GOALS):
        self.beta = beta
        self.cols = cols
        self.cat = cat
        self.rho = rho
        self.max_goals = max_goals

    def _mu(self, frame):
        X = _build_exog(frame, self.cat, self.cols).values
        return np.exp(np.clip(X @ self.beta.values, -30, 30))

    def expected_goals(self, team_a, team_b, neutral=True):
        dom = 0 if neutral else 1
        lam_a = float(self._mu(pd.DataFrame(
            {"equipe": [team_a], "adv": [team_b], "domicile": [dom]}))[0])
        lam_b = float(self._mu(pd.DataFrame(
            {"equipe": [team_b], "adv": [team_a], "domicile": [0]}))[0])
        return lam_a, lam_b

    def score_matrix(self, lam_a, lam_b):
        """Matrice P(score) avec correction Dixon-Coles sur les 4 petits scores."""
        k = np.arange(self.max_goals + 1)
        M = np.outer(poisson.pmf(k, lam_a), poisson.pmf(k, lam_b))
        if self.rho != 0.0:
            # axe 0 = buts equipe_a (domicile), axe 1 = buts equipe_b (exterieur)
            M[0, 0] *= 1.0 - lam_a * lam_b * self.rho
            M[1, 0] *= 1.0 + lam_b * self.rho
            M[0, 1] *= 1.0 + lam_a * self.rho
            M[1, 1] *= 1.0 - self.rho
        # Renormalisation : absorbe la troncature (buts > max_goals) et la
        # correction Dixon-Coles -> p1 + pN + p2 = 1 exactement.
        M /= M.sum()
        return M

    def predict(self, team_a, team_b, neutral=True):
        lam_a, lam_b = self.expected_goals(team_a, team_b, neutral)
        M = self.score_matrix(lam_a, lam_b)
        i, j = np.unravel_index(M.argmax(), M.shape)
        return {
            "lam_a": lam_a, "lam_b": lam_b,
            "p1": float(np.tril(M, -1).sum()),   # A marque plus que B
            "pN": float(np.trace(M)),            # nul
            "p2": float(np.triu(M, 1).sum()),    # B marque plus que A
            "score": (int(i), int(j)),
        }


def _estimate_rho(train, model, weights):
    """Estime ρ (Dixon-Coles) par MV 1-D, sur les seuls matchs à petit score.

    Seul le facteur tau dépend de ρ ; maximiser la vraisemblance DC revient donc
    à maximiser sum(w * log tau) sur les matchs où les deux scores valent 0 ou 1.
    """
    lam = model._mu(pd.DataFrame({
        "equipe": train["home_team"].values, "adv": train["away_team"].values,
        "domicile": np.where(train["neutral"], 0, 1)}))
    mu = model._mu(pd.DataFrame({
        "equipe": train["away_team"].values, "adv": train["home_team"].values,
        "domicile": np.zeros(len(train), dtype=int)}))
    x = train["home_score"].to_numpy(int)
    y = train["away_score"].to_numpy(int)
    w = np.asarray(weights, dtype=float)

    low = (x <= 1) & (y <= 1)
    lam, mu, x, y, w = lam[low], mu[low], x[low], y[low], w[low]
    m00 = (x == 0) & (y == 0); m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1); m11 = (x == 1) & (y == 1)

    def neg_ll(rho):
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m11] = 1.0 - rho
        if np.any(tau <= 0):
            return 1e10
        return -np.sum(w * np.log(tau))

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


def fit_model(train, *, date_ref, half_life=HALF_LIFE_DAYS, friendly_weight=FRIENDLY_WEIGHT,
              lambda_ridge=LAMBDA_RIDGE, max_goals=MAX_GOALS, estimate_rho=True):
    """Ajuste attaque/défense/domicile par MV régularisée (Ridge), puis ρ si demandé."""
    xi = np.log(2) / half_life
    delta = (date_ref - train["date"]).dt.days
    w_time = np.exp(-xi * delta)
    w_imp = np.where(train["tournament"] == "Friendly", friendly_weight, 1.0)
    weights = w_time * w_imp

    dom = pd.DataFrame({
        "buts": train["home_score"].astype(int),
        "equipe": train["home_team"], "adv": train["away_team"],
        "domicile": np.where(train["neutral"], 0, 1), "poids": weights})
    ext = pd.DataFrame({
        "buts": train["away_score"].astype(int),
        "equipe": train["away_team"], "adv": train["home_team"],
        "domicile": 0, "poids": weights})
    long = pd.concat([dom, ext], ignore_index=True)

    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    cat = pd.CategoricalDtype(categories=teams)
    X = _build_exog(long, cat)
    cols = X.columns
    Xmat = X.values
    y = long["buts"].to_numpy(float)
    wv = long["poids"].to_numpy(float)

    pen = np.ones(Xmat.shape[1]); pen[0] = 0.0; pen[1] = 0.0   # libres : const, domicile

    def nll(b):
        eta = np.clip(Xmat @ b, -30, 30); mu = np.exp(eta)
        return np.sum(wv * (mu - y * eta)) + lambda_ridge * np.sum(pen * b * b)

    def grad(b):
        eta = np.clip(Xmat @ b, -30, 30); mu = np.exp(eta)
        return Xmat.T @ (wv * (mu - y)) + 2 * lambda_ridge * pen * b

    b0 = np.zeros(Xmat.shape[1]); b0[0] = np.log(max(y.mean(), 0.1))
    opt = minimize(nll, b0, jac=grad, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
    beta = pd.Series(opt.x, index=cols)

    model = FittedModel(beta, cols, cat, rho=0.0, max_goals=max_goals)
    if estimate_rho:
        model.rho = _estimate_rho(train, model, weights)
    return model
