# Dixon-Coles + avantage hôte + validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter la correction Dixon-Coles et l'avantage hôte au modèle de pronostics CDM 2026, avec une validation hors-échantillon comparant base vs Dixon-Coles.

**Architecture :** Extraire le modèle dans un module partagé `wc_model.py` (chargement, ajustement Poisson, estimation de ρ, prédiction corrigée). `predict_wc2026.py` et `validate.py` deviennent de minces scripts qui consomment ce module — ce qui garantit que la validation et les pronostics utilisent exactement le même code d'ajustement.

**Tech Stack :** Python 3.13 (venv `.venv`), pandas, numpy, scipy, pytest. Dataset `martj42/international_results`.

Spec : [docs/superpowers/specs/2026-06-09-dixon-coles-host-advantage-design.md](../specs/2026-06-09-dixon-coles-host-advantage-design.md)

## Structure des fichiers

| Fichier | Responsabilité |
|---|---|
| `wc_model.py` | Module : `load_data`, `make_train`, `fit_model`, classe `FittedModel` (prédiction Dixon-Coles). |
| `predict_wc2026.py` | Script mince : ajuste sur tout l'historique, prédit les 72 fixtures avec leur vrai `neutral`, écrit le CSV. |
| `validate.py` | Script de validation : hold-out récent, base vs Dixon-Coles, tableau comparatif. |
| `tests/test_wc_model.py` | Tests unitaires sur données synthétiques (rapides, déterministes). |

**Important — commandes :** toujours utiliser l'interpréteur du venv : `.venv/bin/python` et `.venv/bin/pytest`. Ne jamais utiliser `python3` global (pas de pandas).

---

### Task 1: Module `wc_model.py` — chargement + ajustement Poisson (sans Dixon-Coles)

**Goal:** Extraire la logique existante dans un module testable produisant un modèle de Poisson (ρ=0), avec tests unitaires sur données synthétiques.

**Files:**
- Create: `wc_model.py`
- Create: `tests/test_wc_model.py`

**Acceptance Criteria:**
- [ ] `wc_model.fit_model(train, date_ref=..., estimate_rho=False)` renvoie un `FittedModel` avec `rho == 0.0`.
- [ ] `FittedModel.predict(a, b)` renvoie un dict `{lam_a, lam_b, p1, pN, p2, score}` avec `p1+pN+p2 ≈ 1`.
- [ ] Une équipe forte (synthétique) bat une équipe faible : `p1 > p2`.
- [ ] `pytest tests/test_wc_model.py -v` passe (3 tests).

**Verify:** `.venv/bin/pytest tests/test_wc_model.py -v` → 3 passed

**Steps:**

- [ ] **Step 1: Installer pytest dans le venv**

```bash
.venv/bin/pip install --quiet pytest
```

- [ ] **Step 2: Écrire les tests (ils échoueront — module absent)**

Créer `tests/test_wc_model.py` :

```python
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
```

- [ ] **Step 3: Lancer les tests pour les voir échouer**

Run: `.venv/bin/pytest tests/test_wc_model.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'wc_model'`

- [ ] **Step 4: Écrire le module `wc_model.py`**

Créer `wc_model.py` :

```python
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
    """STUB (Task 2) : Poisson indépendant pour l'instant."""
    return 0.0


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
```

- [ ] **Step 5: Lancer les tests pour les voir passer**

Run: `.venv/bin/pytest tests/test_wc_model.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit** (git désactivé pour ce projet — sauter cette étape, le suivi se fait via les tâches)

---

### Task 2: Estimation de ρ (Dixon-Coles) dans `wc_model.py`

**Goal:** Implémenter `_estimate_rho` (optimisation 1-D bornée sur la vraisemblance Dixon-Coles) pour activer la correction des petits scores.

**Files:**
- Modify: `wc_model.py` (remplacer le stub `_estimate_rho`)
- Modify: `tests/test_wc_model.py` (ajouter les tests Dixon-Coles)

**Acceptance Criteria:**
- [ ] `fit_model(..., estimate_rho=True)` renvoie un `rho` fini dans `[-0.2, 0.2]`.
- [ ] Avec un `rho != 0` imposé, `predict` renvoie toujours `p1+pN+p2 ≈ 1` (matrice renormalisée) et la proba de nul change vs `rho=0`.
- [ ] `pytest tests/test_wc_model.py -v` passe (5 tests).

**Verify:** `.venv/bin/pytest tests/test_wc_model.py -v` → 5 passed

**Steps:**

- [ ] **Step 1: Ajouter les tests Dixon-Coles (ils échoueront — stub renvoie 0)**

Ajouter à la fin de `tests/test_wc_model.py` :

```python
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
```

- [ ] **Step 2: Lancer pour voir échouer**

Run: `.venv/bin/pytest tests/test_wc_model.py::test_rho_estimated_finite_in_range -v`
Expected: FAIL (`assert m.rho != 0.0` — le stub renvoie 0.0)

- [ ] **Step 3: Remplacer le stub `_estimate_rho` par l'implémentation réelle**

Dans `wc_model.py`, remplacer la fonction stub :

```python
def _estimate_rho(train, model, weights):
    """STUB (Task 2) : Poisson indépendant pour l'instant."""
    return 0.0
```

par :

```python
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
```

- [ ] **Step 4: Lancer tous les tests pour les voir passer**

Run: `.venv/bin/pytest tests/test_wc_model.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit** (git désactivé — sauter)

---

### Task 3: Réécrire `predict_wc2026.py` (module + avantage hôte réel)

**Goal:** Script mince qui ajuste avec Dixon-Coles sur tout l'historique et prédit les 72 fixtures en lisant leur vrai `neutral` (avantage hôte automatique pour les 9 matchs des pays hôtes).

**Files:**
- Modify: `predict_wc2026.py` (réécriture complète)

**Acceptance Criteria:**
- [ ] Le script s'exécute et régénère `pronostics_cdm2026.csv` avec 72 lignes.
- [ ] `model.rho` non nul est imprimé.
- [ ] Les 9 matchs `neutral=FALSE` montrent un avantage hôte : pour ces matchs, `buts_attendus_A` est strictement supérieur à ce que donnerait un calcul en terrain neutre (vérifié par un contrôle imprimé).
- [ ] Le pays hôte est bien `home_team` dans les 9 lignes `neutral=FALSE` (vérifié par assertion dans le script).

**Verify:** `.venv/bin/python predict_wc2026.py` → imprime `rho`, écrit 72 pronostics, et un bloc « Contrôle avantage hôte » montrant 9 matchs avec écart positif.

**Steps:**

- [ ] **Step 1: Réécrire `predict_wc2026.py`**

Remplacer tout le contenu de `predict_wc2026.py` par :

```python
"""
Pronostics de la phase de poule de la Coupe du Monde 2026.

Modèle de Poisson Maher / Dixon-Coles (module wc_model) :
  - ajustement attaque/défense/domicile par MV régularisée Ridge,
  - pondération temporelle (demi-vie 2 ans), amicaux à poids réduit,
  - correction Dixon-Coles des petits scores (rho),
  - avantage hôte appliqué aux matchs où USA/Mexique/Canada jouent chez eux
    (lu directement depuis la colonne `neutral` du dataset).
"""
import numpy as np
import pandas as pd
import wc_model as wm

DATE_REF = pd.Timestamp("2026-06-11")   # début CDM : référence du decay

# 1) Données + fixtures à prédire (scores NA)
df = wm.load_data()
fixtures = df[(df["tournament"] == "FIFA World Cup")
              & (df["date"].dt.year == 2026)
              & (df["home_score"].isna())].copy()

# Sanity : dans les matchs non neutres, le pays hôte est bien l'équipe à domicile
hosts = {"United States", "Mexico", "Canada"}
host_games = fixtures[~fixtures["neutral"].astype(bool)]
assert set(host_games["home_team"]).issubset(hosts), \
    "Hypothese cassee : un match non-neutre n'a pas un hote comme home_team"

# 2) Entraînement (toutes les équipes de la CDM conservées) + ajustement Dixon-Coles
wc_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
train = wm.make_train(df, date_ref=DATE_REF, keep_teams=wc_teams)
model = wm.fit_model(train, date_ref=DATE_REF, estimate_rho=True)
print("rho (Dixon-Coles) = %.4f | avantage terrain : x%.2f de buts a domicile"
      % (model.rho, np.exp(model.beta["domicile"])))

# 3) Prédiction des 72 matchs (avantage hôte via le vrai `neutral`)
rows, controle = [], []
for _, m in fixtures.sort_values("date").iterrows():
    a, b = m["home_team"], m["away_team"]
    neutral = bool(m["neutral"])
    r = model.predict(a, b, neutral=neutral)
    if not neutral:  # contrôle : comparer au même match en terrain neutre
        r_neutre = model.predict(a, b, neutral=True)
        controle.append((a, b, r["lam_a"], r_neutre["lam_a"]))
    issues = {"1": r["p1"], "N": r["pN"], "2": r["p2"]}
    prono = max(issues, key=issues.get)
    rows.append({
        "date": m["date"].date(), "equipe_A": a, "equipe_B": b,
        "buts_attendus_A": round(r["lam_a"], 2), "buts_attendus_B": round(r["lam_b"], 2),
        "P_victoire_A": round(100 * r["p1"], 1), "P_nul": round(100 * r["pN"], 1),
        "P_victoire_B": round(100 * r["p2"], 1),
        "score_probable": f"{r['score'][0]}-{r['score'][1]}", "prono_1N2": prono,
    })

res = pd.DataFrame(rows)
res.to_csv("pronostics_cdm2026.csv", index=False)
print("\n%d pronostics enregistres dans pronostics_cdm2026.csv" % len(res))

print("\nControle avantage hote (buts attendus hote : avec avantage vs neutre) :")
for a, b, lam_dom, lam_neu in controle:
    print("  %-15s vs %-15s : %.2f  vs  %.2f  (+%.2f)"
          % (a, b, lam_dom, lam_neu, lam_dom - lam_neu))
assert len(controle) == 9 and all(d > n for _, _, d, n in controle), \
    "L'avantage hote n'a pas ete applique aux 9 matchs attendus"
```

- [ ] **Step 2: Lancer le script**

Run: `.venv/bin/python predict_wc2026.py`
Expected: imprime `rho (Dixon-Coles) = -0.0xxx`, `72 pronostics enregistres`, puis un bloc « Controle avantage hote » de 9 lignes avec écart `(+0.xx)` positif. Pas d'`AssertionError`.

- [ ] **Step 3: Vérifier le CSV régénéré**

Run: `.venv/bin/python -c "import pandas as pd; d=pd.read_csv('pronostics_cdm2026.csv'); print(len(d)); print(d.head())"`
Expected: 72 lignes.

- [ ] **Step 4: Commit** (git désactivé — sauter)

---

### Task 4: Script de validation `validate.py` (base vs Dixon-Coles)

**Goal:** Mesurer hors-échantillon si Dixon-Coles améliore les prédictions, via un tableau comparatif exactitude / log-loss / calibration des nuls.

**Files:**
- Create: `validate.py`

**Acceptance Criteria:**
- [ ] Le script ajuste base (`estimate_rho=False`) et Dixon-Coles (`estimate_rho=True`) sur le **même** train (matchs avant `TEST_START`).
- [ ] Évalue sur les matchs joués depuis `TEST_START`, restreints aux équipes connues du modèle.
- [ ] Imprime un tableau avec, pour chaque modèle : exactitude 1/N/2, log-loss, P(nul) prédite moyenne, et la fréquence réelle de nuls.
- [ ] Le nombre de matchs test imprimé est > 1000.

**Verify:** `.venv/bin/python validate.py` → tableau comparatif `base` vs `Dixon-Coles` + fréquence réelle de nuls.

**Steps:**

- [ ] **Step 1: Écrire `validate.py`**

Créer `validate.py` :

```python
"""
Validation hors-échantillon : le modèle Dixon-Coles bat-il le Poisson indépendant ?

On cache tous les matchs joués depuis TEST_START, on ajuste base et Dixon-Coles sur
le même historique antérieur, et on compare exactitude 1/N/2, log-loss et calibration
des nuls. Les deux modèles utilisent le `neutral` réel des matchs test, donc la
comparaison isole l'apport de la correction Dixon-Coles.
"""
import numpy as np
import pandas as pd
import wc_model as wm

TEST_START = pd.Timestamp("2024-01-01")

df = wm.load_data()
train = wm.make_train(df, date_ref=TEST_START)
known = set(train["home_team"]) | set(train["away_team"])

test = df[(df["date"] >= TEST_START) & df["home_score"].notna()].copy()
test = test[test["home_team"].isin(known) & test["away_team"].isin(known)].copy()
print("Train : %d matchs (< %s) | Test : %d matchs (>= %s)"
      % (len(train), TEST_START.date(), len(test), TEST_START.date()))


def evaluate(model, test):
    correct = 0
    ll = 0.0
    pred_draw = []
    is_draw = []
    for _, m in test.iterrows():
        r = model.predict(m["home_team"], m["away_team"], neutral=bool(m["neutral"]))
        p = np.array([r["p1"], r["pN"], r["p2"]])
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        outcome = 0 if hs > as_ else (1 if hs == as_ else 2)
        correct += int(p.argmax() == outcome)
        ll += -np.log(max(p[outcome], 1e-12))
        pred_draw.append(p[1])
        is_draw.append(outcome == 1)
    n = len(test)
    return {
        "exactitude": correct / n,
        "log_loss": ll / n,
        "p_nul_moyen": float(np.mean(pred_draw)),
    }, float(np.mean(is_draw))


base = wm.fit_model(train, date_ref=TEST_START, estimate_rho=False)
dc = wm.fit_model(train, date_ref=TEST_START, estimate_rho=True)
print("rho estime (Dixon-Coles) = %.4f" % dc.rho)

res_base, freq_nul = evaluate(base, test)
res_dc, _ = evaluate(dc, test)

print("\n%-14s | %10s | %10s | %12s" % ("Modele", "Exactitude", "Log-loss", "P(nul) moy."))
print("-" * 56)
for nom, r in [("Base (Poisson)", res_base), ("Dixon-Coles", res_dc)]:
    print("%-14s | %9.1f%% | %10.4f | %11.1f%%"
          % (nom, 100 * r["exactitude"], r["log_loss"], 100 * r["p_nul_moyen"]))
print("-" * 56)
print("Frequence reelle de nuls sur le test : %.1f%%" % (100 * freq_nul))
print("\nLecture : log-loss plus bas = mieux. Dixon-Coles devrait rapprocher")
print("le 'P(nul) moyen' de la frequence reelle de nuls (meilleure calibration).")
```

- [ ] **Step 2: Lancer la validation**

Run: `.venv/bin/python validate.py`
Expected: ligne `Train : ... | Test : N matchs` avec N > 1000, `rho estime`, puis un tableau à 2 lignes (Base, Dixon-Coles) et la fréquence réelle de nuls.

- [ ] **Step 3: Commit** (git désactivé — sauter)

---

## Notes d'exécution

- **Git désactivé** pour ce projet (choix utilisateur) : ignorer toutes les étapes « Commit ». Le suivi se fait via la liste de tâches.
- **Toujours** invoquer `.venv/bin/python` / `.venv/bin/pytest`.
- L'ajustement sur l'historique complet prend quelques secondes ; les tests synthétiques sont quasi instantanés.
- Valeur attendue de `rho` : petite et négative (~ -0,03 à -0,15 selon le dataset).
