# Simulation Monte-Carlo de la CDM 2026 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simuler par Monte-Carlo l'intégralité de la CDM 2026 (poule + phase finale) avec le modèle Dixon-Coles existant, pour estimer P(titre) et les buts totaux attendus par équipe.

**Architecture :** Données du tableau figées dans `bracket_2026.py` (recherche web validée), logique pure du tournoi dans `tournament.py` (classement, sélection des 3es, échantillonnage de match, construction du tableau, simulation complète — testable sur entrées synthétiques), et un driver `simulate_wc2026.py` qui ajuste le modèle, pré-calcule les distributions de score, lance N simulations et agrège.

**Tech Stack :** Python 3.13 (venv `.venv`), numpy, pandas, scipy, pytest. Réutilise `wc_model.py`.

Spec : [docs/superpowers/specs/2026-06-09-monte-carlo-simulation-design.md](../specs/2026-06-09-monte-carlo-simulation-design.md)

## Structure des fichiers

| Fichier | Responsabilité |
|---|---|
| `bracket_2026.py` | Données figées : `GROUPS` (lettre A-L → 4 équipes), `R32` (16 paires de créneaux, ordre officiel du tableau). Validé contre les groupes reconstruits du dataset. |
| `tournament.py` | Logique pure : `make_match_dist`, `simulate_match`, `rank_group`, `select_best_thirds`, `build_bracket`, `simulate_tournament`. |
| `simulate_wc2026.py` | Driver : ajuste le modèle, reconstruit/valide les groupes, pré-calcule les distributions, lance N sims, agrège, écrit `simulation_cdm2026.csv`. |
| `tests/test_tournament.py` | Tests unitaires sur entrées synthétiques. |

**Schéma de données partagé (fixé ici, utilisé par toutes les tâches) :**
- `GROUPS: dict[str, list[str]]` — 12 clés `"A"`..`"L"`, chacune 4 noms d'équipes.
- `R32: list[tuple[str, str]]` — 16 paires de **créneaux**, dans l'ordre officiel du tableau (l'arbre au-delà des 32es est implicite : le vainqueur du match `2k` affronte le vainqueur du match `2k+1`). Un créneau est :
  - `"1X"` / `"2X"` (X = lettre de groupe) : 1er / 2e du groupe X,
  - une chaîne commençant par `"T"` (`"T1"`..`"T8"`) : créneau d'un des 8 meilleurs 3es.
- **Distribution de match** `dist = (cum, p1, p2)` : `cum` = probas cumulées (121,) de la matrice de score aplatie (11×11, `idx = i*11 + j` → i buts équipe A, j buts équipe B) ; `p1` = P(A gagne), `p2` = P(B gagne).
- **Niveaux atteints** (entiers) : 0 éliminé en poule · 1 qualifié (32es) · 2 en 8es · 3 en quarts · 4 en demis · 5 en finale · 6 champion.

**Commandes :** toujours `.venv/bin/python` / `.venv/bin/pytest`. Git désactivé → ignorer les commits.

---

### Task 1: `bracket_2026.py` — données officielles du tableau (recherche web + validation)

**Goal:** Figer dans un module les 12 groupes officiels (lettres A-L) et le squelette officiel des 32es, et prouver leur cohérence avec les 72 affiches du dataset.

**Files:**
- Create: `bracket_2026.py`
- Create: `tests/test_bracket_2026.py`

**Acceptance Criteria:**
- [ ] `bracket_2026.GROUPS` a 12 clés `"A"`..`"L"`, chacune avec exactement 4 équipes ; l'union des 48 équipes est **identique** à l'ensemble des équipes des 72 fixtures.
- [ ] Pour chaque groupe, les 4 équipes forment bien une composante de matchs du dataset (les 6 paires intra-groupe existent dans les fixtures).
- [ ] `bracket_2026.R32` a 16 paires ; tous les créneaux `"1X"/"2X"` référencent une lettre de `GROUPS` ; il y a exactement 8 créneaux commençant par `"T"`, tous distincts (`T1`..`T8`).
- [ ] `pytest tests/test_bracket_2026.py -v` passe.

**Verify:** `.venv/bin/pytest tests/test_bracket_2026.py -v` → passed

**Steps:**

- [ ] **Step 1: Reconstruire les groupes depuis le dataset (référence de validation)**

Exécuter pour obtenir les 12 groupes (4 équipes) tels qu'encodés dans les fixtures — sert de vérité terrain :

```bash
.venv/bin/python - <<'PY'
import wc_model as wm
from collections import defaultdict
df = wm.load_data()
fx = df[(df.tournament=="FIFA World Cup") & (df.date.dt.year==2026) & (df.home_score.isna())]
parent={}
def find(x):
    parent.setdefault(x,x)
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
for _,m in fx.iterrows(): parent[find(m.home_team)]=find(m.away_team)
g=defaultdict(list)
for t in set(fx.home_team)|set(fx.away_team): g[find(t)].append(t)
for c in sorted(g.values(), key=lambda s:min(s)): print(sorted(c))
PY
```

- [ ] **Step 2: Rechercher les données officielles**

Avec l'outil WebSearch, récupérer pour la Coupe du Monde 2026 :
1. la composition officielle des groupes **A à L** (tirage du 5 décembre 2025),
2. le **squelette des 32es de finale** (quel 1er/2e de groupe occupe quel créneau, et l'ordre des 16 matchs dans le tableau).

Recouper au moins deux sources (ex. site FIFA, Wikipédia « 2026 FIFA World Cup knockout stage »). Associer chaque groupe reconstruit (Step 1) à sa lettre officielle.

- [ ] **Step 3: CHECKPOINT utilisateur**

Présenter à l'utilisateur la correspondance « lettre officielle → 4 équipes » et le squelette des 32es, et **attendre sa validation** avant d'écrire le module (exigence du design). Corriger si l'utilisateur signale une erreur.

- [ ] **Step 4: Écrire `bracket_2026.py`**

Écrire le module avec les valeurs validées. Forme attendue (remplir avec les données officielles validées au Step 3) :

```python
"""
Données officielles du tableau de la Coupe du Monde 2026 (tirage du 05/12/2025).

GROUPS : lettre de groupe -> 4 équipes (noms tels qu'ils apparaissent dans le
         dataset martj42, ex. "United States", "South Korea").
R32    : 16 matchs des 32es dans l'ordre officiel du tableau. Chaque créneau est
         "1X"/"2X" (1er/2e du groupe X) ou "T1".."T8" (un des 8 meilleurs 3es).
         L'arbre au-delà : vainqueur(match 2k) vs vainqueur(match 2k+1).
"""

GROUPS = {
    "A": ["Mexico", "...", "...", "..."],
    # ... B..L renseignés avec les données officielles validées
}

R32 = [
    ("1A", "T1"),
    ("1C", "2D"),
    # ... 16 paires au total, dans l'ordre officiel du tableau
]
```

- [ ] **Step 5: Écrire et lancer les tests de cohérence**

Créer `tests/test_bracket_2026.py` :

```python
"""Cohérence des données du tableau 2026 avec les fixtures du dataset."""
from collections import defaultdict
import wc_model as wm
import bracket_2026 as bk


def _fixtures():
    df = wm.load_data()
    return df[(df.tournament == "FIFA World Cup")
              & (df.date.dt.year == 2026)
              & (df.home_score.isna())]


def test_groups_cover_all_48_fixture_teams():
    fx = _fixtures()
    fixture_teams = set(fx.home_team) | set(fx.away_team)
    group_teams = set(t for teams in bk.GROUPS.values() for t in teams)
    assert len(bk.GROUPS) == 12
    assert all(len(v) == 4 for v in bk.GROUPS.values())
    assert group_teams == fixture_teams


def test_each_group_is_a_real_fixture_clique():
    fx = _fixtures()
    pairs = set(frozenset((m.home_team, m.away_team)) for _, m in fx.iterrows())
    for teams in bk.GROUPS.values():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                assert frozenset((teams[i], teams[j])) in pairs


def test_r32_skeleton_well_formed():
    assert len(bk.R32) == 16
    slots = [s for pair in bk.R32 for s in pair]
    thirds = sorted(s for s in slots if s.startswith("T"))
    assert thirds == [f"T{i}" for i in range(1, 9)]
    for s in slots:
        if not s.startswith("T"):
            assert s[0] in ("1", "2") and s[1:] in bk.GROUPS
```

Run: `.venv/bin/pytest tests/test_bracket_2026.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit** (git désactivé — sauter)

---

### Task 2: `tournament.py` — primitives (classement, 3es, échantillonnage de match)

**Goal:** Implémenter les briques pures testables : distribution de match, échantillonnage (avec résolution des nuls en phase finale), classement de groupe, sélection des meilleurs 3es.

**Files:**
- Create: `tournament.py`
- Create: `tests/test_tournament.py`

**Acceptance Criteria:**
- [ ] `make_match_dist(model, a, b, neutral)` renvoie `(cum, p1, p2)` avec `cum[-1] ≈ 1`.
- [ ] `simulate_match` sur une distribution dégénérée (victoire certaine de A) renvoie toujours A vainqueur ; en phase finale, un nul forcé est tranché (jamais `None`).
- [ ] `rank_group` ordonne par points, puis diff. de buts, puis buts marqués.
- [ ] `select_best_thirds` retient les 8 meilleurs 3es selon le même ordre.
- [ ] `pytest tests/test_tournament.py -v` passe.

**Verify:** `.venv/bin/pytest tests/test_tournament.py -v` → passed

**Steps:**

- [ ] **Step 1: Écrire les tests des primitives**

Créer `tests/test_tournament.py` :

```python
"""Tests des primitives de tournoi sur entrées synthétiques."""
import numpy as np
import tournament as tn


def _dist_a_sure():
    # 11x11 aplati : toute la masse sur le score 1-0 (A gagne à coup sûr).
    probs = np.zeros(121)
    probs[1 * 11 + 0] = 1.0
    return (np.cumsum(probs), 1.0, 0.0)


def _dist_draw_sure(p1=0.7, p2=0.3):
    # toute la masse sur 0-0 ; p1/p2 servent à trancher en phase finale.
    probs = np.zeros(121)
    probs[0] = 1.0
    return (np.cumsum(probs), p1, p2)


def test_simulate_match_group_returns_goals():
    rng = np.random.default_rng(0)
    ga, gb, a_wins = tn.simulate_match(rng, _dist_a_sure(), knockout=False)
    assert (ga, gb) == (1, 0)
    assert a_wins is True


def test_simulate_match_group_draw_is_none():
    rng = np.random.default_rng(0)
    ga, gb, a_wins = tn.simulate_match(rng, _dist_draw_sure(), knockout=False)
    assert (ga, gb) == (0, 0)
    assert a_wins is None


def test_simulate_match_knockout_resolves_draw():
    rng = np.random.default_rng(0)
    # p1=1.0 => A gagne toujours la séance de tirs au but
    outcomes = [tn.simulate_match(rng, _dist_draw_sure(p1=1.0, p2=0.0), knockout=True)[2]
                for _ in range(20)]
    assert all(w is True for w in outcomes)


def test_rank_group_orders_by_points_then_gd_then_gf():
    # stats[team] = [points, gf, ga]
    stats = {
        "W": [9, 7, 1],   # 1er : plus de points
        "X": [4, 5, 4],   # 2e : 4 pts, diff +1
        "Y": [4, 6, 6],   # 3e : 4 pts, diff 0
        "Z": [1, 2, 9],   # 4e
    }
    rng = np.random.default_rng(0)
    assert tn.rank_group(["W", "X", "Y", "Z"], stats, rng) == ["W", "X", "Y", "Z"]


def test_select_best_thirds_takes_top_8():
    # 12 troisièmes (team, points, gd, gf, group) ; on garde les 8 meilleurs.
    thirds = [(f"G{i}", i, 0, 0, chr(65 + i)) for i in range(12)]  # points croissants 0..11
    rng = np.random.default_rng(0)
    kept = tn.select_best_thirds(thirds, rng)
    assert len(kept) == 8
    # les 8 meilleurs sont ceux à points 11..4
    assert set(kept) == {f"G{i}" for i in range(4, 12)}
```

- [ ] **Step 2: Lancer pour voir échouer**

Run: `.venv/bin/pytest tests/test_tournament.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'tournament'`)

- [ ] **Step 3: Écrire les primitives dans `tournament.py`**

Créer `tournament.py` :

```python
"""
Logique pure de simulation du tournoi (sans I/O), pilotée par un RNG numpy.

Distribution de match : (cum, p1, p2) où cum = probas cumulées (121,) de la
matrice de score 11x11 aplatie (idx = i*11 + j ; i buts A, j buts B), p1 = P(A
gagne), p2 = P(B gagne). Voir bracket_2026 pour le schéma GROUPS / R32.
"""
import numpy as np

NB = 11  # 0..10 buts par équipe (taille de la matrice du modèle)


def make_match_dist(model, team_a, team_b, neutral=True):
    """Construit (cum, p1, p2) à partir de la matrice Dixon-Coles du modèle."""
    lam_a, lam_b = model.expected_goals(team_a, team_b, neutral=neutral)
    M = model.score_matrix(lam_a, lam_b)              # (11, 11), somme = 1
    cum = np.cumsum(M.ravel())
    p1 = float(np.tril(M, -1).sum())                  # A marque plus que B
    p2 = float(np.triu(M, 1).sum())                   # B marque plus que A
    return (cum, p1, p2)


def simulate_match(rng, dist, knockout):
    """Tire un score. Renvoie (buts_a, buts_b, a_wins).

    a_wins : True/False en cas de résultat décisif ; None si nul en poule.
    En phase finale, un nul est tranché par 'tirs au but' ~ Bernoulli(p1/(p1+p2)).
    """
    cum, p1, p2 = dist
    idx = int(np.searchsorted(cum, rng.random()))
    ga, gb = divmod(idx, NB)
    if ga > gb:
        return ga, gb, True
    if gb > ga:
        return ga, gb, False
    if not knockout:
        return ga, gb, None
    total = p1 + p2
    a_wins = rng.random() < (p1 / total if total > 0 else 0.5)
    return ga, gb, bool(a_wins)


def rank_group(teams, stats, rng):
    """Ordonne les équipes (meilleure d'abord). stats[t] = [points, gf, ga].
    Départage : points -> diff. de buts -> buts marqués -> tirage aléatoire."""
    jitter = {t: rng.random() for t in teams}
    return sorted(teams, key=lambda t: (
        stats[t][0], stats[t][1] - stats[t][2], stats[t][1], jitter[t]
    ), reverse=True)


def select_best_thirds(thirds, rng):
    """thirds : liste de (team, points, gd, gf, group). Renvoie les 8 meilleurs
    (points -> gd -> gf -> aléatoire)."""
    jitter = {t[0]: rng.random() for t in thirds}
    ordered = sorted(thirds, key=lambda x: (x[1], x[2], x[3], jitter[x[0]]), reverse=True)
    return [t[0] for t in ordered[:8]]
```

- [ ] **Step 4: Lancer pour voir passer**

Run: `.venv/bin/pytest tests/test_tournament.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit** (git désactivé — sauter)

---

### Task 3: `tournament.py` — construction du tableau & simulation complète

**Goal:** Ajouter la construction du tableau des 32es (placement des 1ers/2es/3es) et `simulate_tournament`, qui simule un tournoi entier et renvoie, par équipe, le niveau atteint, les buts marqués et le nombre de matchs joués.

**Files:**
- Modify: `tournament.py` (ajouter `build_bracket`, `simulate_tournament` et l'aide `_assign_thirds`)
- Modify: `tests/test_tournament.py` (ajouter les tests du tableau et de la simulation)

**Acceptance Criteria:**
- [ ] `build_bracket` place les 1ers/2es selon le squelette et affecte les 8 troisièmes en évitant qu'un 3e affronte d'entrée une équipe de son propre groupe (quand c'est possible).
- [ ] `simulate_tournament` avec des distributions déterministes (la plus forte gagne toujours) donne niveau 6 (champion) à cette équipe et niveau ≥1 à 32 équipes exactement.
- [ ] Les buts d'une équipe éliminée en poule = somme de ses 3 matchs de groupe.
- [ ] `pytest tests/test_tournament.py -v` passe (tests primitives + nouveaux).

**Verify:** `.venv/bin/pytest tests/test_tournament.py -v` → passed

**Steps:**

- [ ] **Step 1: Ajouter les tests du tableau et de la simulation**

Ajouter à la fin de `tests/test_tournament.py` :

```python
import bracket_2026 as bk
import wc_model as wm


def _toy_world():
    """Construit un mini-univers déterministe : 12 groupes A-L de 4 équipes,
    où l'équipe est d'autant plus forte que son indice global est petit. Les
    distributions sont dégénérées (la mieux classée gagne toujours, 1-0)."""
    groups = {}
    team_rank = {}
    k = 0
    for L in "ABCDEFGHIJKL":
        teams = [f"{L}{i}" for i in range(4)]
        groups[L] = teams
        for t in teams:
            team_rank[t] = k
            k += 1

    def dist(a, b):
        probs = np.zeros(121)
        if team_rank[a] < team_rank[b]:
            probs[1 * 11 + 0] = 1.0   # A gagne 1-0
            p1, p2 = 1.0, 0.0
        else:
            probs[0 * 11 + 1] = 1.0   # B gagne 0-1
            p1, p2 = 0.0, 1.0
        return (np.cumsum(probs), p1, p2)

    # fixtures de poule : round-robin de chaque groupe (6 matchs)
    group_fixtures = {}
    for L, teams in groups.items():
        gf = []
        for i in range(4):
            for j in range(i + 1, 4):
                gf.append((teams[i], teams[j]))
        group_fixtures[L] = gf

    # distributions : groupe (par fixture) et neutre (tout couple ordonné)
    group_dist = {L: {m: dist(*m) for m in group_fixtures[L]} for L in groups}
    all_teams = [t for teams in groups.values() for t in teams]
    neutral_dist = {(a, b): dist(a, b) for a in all_teams for b in all_teams if a != b}
    return groups, group_fixtures, group_dist, neutral_dist


def test_build_bracket_avoids_same_group_third():
    groups, gf, gd, nd = _toy_world()
    rng = np.random.default_rng(0)
    res = tn.simulate_tournament(rng, groups, gf, gd, nd, bk.R32)
    # 32 équipes atteignent au moins le niveau 1 (qualifiées)
    qualified = [t for t, r in res.items() if r["level"] >= 1]
    assert len(qualified) == 32


def test_strongest_team_always_champion():
    groups, gf, gd, nd = _toy_world()
    rng = np.random.default_rng(1)
    res = tn.simulate_tournament(rng, groups, gf, gd, nd, bk.R32)
    champ = max(res, key=lambda t: res[t]["level"])
    assert res["A0"]["level"] == 6        # l'équipe globalement la plus forte
    assert champ == "A0"


def test_group_eliminated_team_goals_equal_three_matches():
    groups, gf, gd, nd = _toy_world()
    rng = np.random.default_rng(2)
    res = tn.simulate_tournament(rng, groups, gf, gd, nd, bk.R32)
    # une équipe classée 4e de son groupe joue exactement 3 matchs
    eliminated = [t for t, r in res.items() if r["level"] == 0]
    assert eliminated, "il doit rester des éliminés de poule"
    for t in eliminated:
        assert res[t]["matches"] == 3
```

- [ ] **Step 2: Lancer pour voir échouer**

Run: `.venv/bin/pytest tests/test_tournament.py -k "bracket or champion or eliminated" -v`
Expected: FAIL (`AttributeError: module 'tournament' has no attribute 'simulate_tournament'`)

- [ ] **Step 3: Ajouter `build_bracket` + `simulate_tournament` à `tournament.py`**

Ajouter à la fin de `tournament.py` :

```python
from collections import defaultdict


def _assign_thirds(third_slots, qual_thirds, forbidden, third_group, rng):
    """Affecte les 8 troisièmes qualifiés aux 8 créneaux 'T' en évitant qu'un
    3e tombe sur une équipe de son groupe (forbidden[slot] = lettre interdite).
    Backtracking déterministe ; repli sans contrainte si aucune solution."""
    slots = list(third_slots)
    teams = list(qual_thirds)

    def backtrack(i, used, acc):
        if i == len(slots):
            return dict(acc)
        for t in teams:
            if t in used:
                continue
            if third_group[t] == forbidden.get(slots[i]):
                continue
            acc.append((slots[i], t)); used.add(t)
            sol = backtrack(i + 1, used, acc)
            if sol is not None:
                return sol
            acc.pop(); used.discard(t)
        return None

    sol = backtrack(0, set(), [])
    if sol is None:  # repli : appariement arbitraire (cas dégénéré rarissime)
        sol = dict(zip(slots, teams))
    return sol


def build_bracket(skeleton, group_rank, qual_thirds, third_group, rng):
    """Transforme le squelette de créneaux en 16 matchs d'équipes réelles.
    group_rank[L] = [1er, 2e, 3e, 4e] ; qual_thirds = 8 équipes ; third_group[t]=lettre."""
    def fixed(slot):
        return group_rank[slot[1]][0 if slot[0] == "1" else 1]

    third_slots = [s for pair in skeleton for s in pair if s.startswith("T")]
    forbidden = {}
    for sa, sb in skeleton:
        for s, other in ((sa, sb), (sb, sa)):
            if s.startswith("T") and not other.startswith("T"):
                forbidden[s] = other[1]  # lettre de l'adversaire fixe
    assign = _assign_thirds(third_slots, qual_thirds, forbidden, third_group, rng)

    def resolve(slot):
        return assign[slot] if slot.startswith("T") else fixed(slot)

    return [(resolve(sa), resolve(sb)) for sa, sb in skeleton]


def simulate_tournament(rng, groups, group_fixtures, group_dist, neutral_dist, skeleton):
    """Simule un tournoi complet. Renvoie {team: {'level', 'goals', 'matches'}}."""
    goals = defaultdict(int)
    matches = defaultdict(int)
    level = {}
    group_rank = {}
    third_group = {}
    thirds = []

    # --- Phase de poule ---
    for L, teams in groups.items():
        stats = {t: [0, 0, 0] for t in teams}   # [points, gf, ga]
        for (h, a) in group_fixtures[L]:
            ga, gb, a_wins = simulate_match(rng, group_dist[L][(h, a)], knockout=False)
            goals[h] += ga; goals[a] += gb
            matches[h] += 1; matches[a] += 1
            stats[h][1] += ga; stats[h][2] += gb
            stats[a][1] += gb; stats[a][2] += ga
            if a_wins is True:
                stats[h][0] += 3
            elif a_wins is False:
                stats[a][0] += 3
            else:
                stats[h][0] += 1; stats[a][0] += 1
        ranked = rank_group(teams, stats, rng)
        group_rank[L] = ranked
        for t in ranked:
            level[t] = 0
        level[ranked[0]] = 1
        level[ranked[1]] = 1
        s = stats[ranked[2]]
        thirds.append((ranked[2], s[0], s[1] - s[2], s[1], L))
        third_group[ranked[2]] = L

    # --- Meilleurs troisièmes ---
    qual_thirds = select_best_thirds(thirds, rng)
    for t in qual_thirds:
        level[t] = 1

    # --- Tableau + phase finale ---
    current = build_bracket(skeleton, group_rank, qual_thirds, third_group, rng)
    reach_level = 2
    while True:
        winners = []
        for (a, b) in current:
            ga, gb, a_wins = simulate_match(rng, neutral_dist[(a, b)], knockout=True)
            goals[a] += ga; goals[b] += gb
            matches[a] += 1; matches[b] += 1
            w = a if a_wins else b
            level[w] = max(level[w], reach_level)
            winners.append(w)
        if len(winners) == 1:
            break
        current = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        reach_level += 1

    return {t: {"level": level[t], "goals": goals[t], "matches": matches[t]} for t in level}
```

- [ ] **Step 4: Lancer toute la suite**

Run: `.venv/bin/pytest tests/test_tournament.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit** (git désactivé — sauter)

---

### Task 4: `simulate_wc2026.py` — driver, agrégation, CSV

**Goal:** Brancher le modèle réel sur la simulation : ajuster, reconstruire/valider les groupes, pré-calculer les distributions, lancer N simulations, agréger en probabilités et buts attendus, écrire le CSV et imprimer un top 15.

**Files:**
- Create: `simulate_wc2026.py`

**Acceptance Criteria:**
- [ ] Le script produit `simulation_cdm2026.csv` avec 48 lignes et les colonnes `equipe, P_titre, P_finale, P_demi, P_quart, P_8e, P_qualifie, buts_totaux_attendus, matchs_joues_attendus`.
- [ ] La somme des `P_titre` sur les 48 équipes vaut 1 (± tolérance Monte-Carlo, vérifiée par assertion `abs(sum-1) < 1e-9` car ce sont des fréquences).
- [ ] Les distributions de groupe utilisent l'avantage hôte (via le `neutral` réel des 9 fixtures) ; la phase finale est en terrain neutre.
- [ ] Le top 15 imprimé met des favoris crédibles en tête (ex. Argentine, Espagne, Brésil, France).
- [ ] `N` (nb de simulations) est une constante en tête de fichier (défaut 20000).

**Verify:** `.venv/bin/python simulate_wc2026.py` → imprime un top 15 + écrit `simulation_cdm2026.csv` (48 lignes).

**Steps:**

- [ ] **Step 1: Écrire `simulate_wc2026.py`**

Créer `simulate_wc2026.py` :

```python
"""
Simulation Monte-Carlo de la Coupe du Monde 2026.

Ajuste le modèle Dixon-Coles (wc_model), reconstruit les groupes et le tableau
(bracket_2026), pré-calcule les distributions de score, puis simule N tournois
complets (tournament.simulate_tournament) pour estimer, par équipe :
  - la probabilité d'atteindre chaque tour et de gagner le titre,
  - les buts totaux attendus et le nombre de matchs joués attendu.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import wc_model as wm
import bracket_2026 as bk
import tournament as tn

N = 20000                              # nombre de simulations
DATE_REF = pd.Timestamp("2026-06-11")
SEED = 42

# 1) Ajustement du modèle (comme predict_wc2026)
df = wm.load_data()
fixtures = df[(df["tournament"] == "FIFA World Cup")
              & (df["date"].dt.year == 2026)
              & (df["home_score"].isna())].copy()
wc_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
train = wm.make_train(df, date_ref=DATE_REF, keep_teams=wc_teams)
model = wm.fit_model(train, date_ref=DATE_REF, estimate_rho=True)
print("Modele ajuste : rho=%.4f | %d equipes | N=%d simulations"
      % (model.rho, len(wc_teams), N))

# 2) Validation : les groupes officiels couvrent bien les 48 équipes des fixtures
assert set(t for teams in bk.GROUPS.values() for t in teams) == wc_teams, \
    "bracket_2026.GROUPS ne correspond pas aux equipes des fixtures"

# 3) Fixtures de poule par groupe (avec le vrai flag neutral pour l'avantage hote)
team_to_group = {t: L for L, teams in bk.GROUPS.items() for t in teams}
group_fixtures = defaultdict(list)
fixture_neutral = {}
for _, m in fixtures.iterrows():
    L = team_to_group[m["home_team"]]
    group_fixtures[L].append((m["home_team"], m["away_team"]))
    fixture_neutral[(m["home_team"], m["away_team"])] = bool(m["neutral"])

# 4) Pré-calcul des distributions de score
#    - groupe : une dist par fixture, avantage hote via neutral reel
#    - neutre : une dist par couple ordonne (phase finale)
group_dist = {L: {pair: tn.make_match_dist(model, pair[0], pair[1],
                                            neutral=fixture_neutral[pair])
                  for pair in pairs}
              for L, pairs in group_fixtures.items()}
teams = sorted(wc_teams)
neutral_dist = {(a, b): tn.make_match_dist(model, a, b, neutral=True)
                for a in teams for b in teams if a != b}
print("Distributions pre-calculees (%d couples neutres)." % len(neutral_dist))

# 5) N simulations
rng = np.random.default_rng(SEED)
reach = defaultdict(lambda: np.zeros(7, dtype=np.int64))  # comptes par niveau atteint (0..6)
sum_goals = defaultdict(float)
sum_matches = defaultdict(float)
for _ in range(N):
    res = tn.simulate_tournament(rng, bk.GROUPS, group_fixtures,
                                 group_dist, neutral_dist, bk.R32)
    for t, r in res.items():
        for lvl in range(r["level"] + 1):
            reach[t][lvl] += 1
        sum_goals[t] += r["goals"]
        sum_matches[t] += r["matches"]

# 6) Agrégation -> probabilités et moyennes
rows = []
for t in teams:
    c = reach[t]
    rows.append({
        "equipe": t,
        "P_titre": c[6] / N,
        "P_finale": c[5] / N,
        "P_demi": c[4] / N,
        "P_quart": c[3] / N,
        "P_8e": c[2] / N,
        "P_qualifie": c[1] / N,
        "buts_totaux_attendus": sum_goals[t] / N,
        "matchs_joues_attendus": sum_matches[t] / N,
    })
res = pd.DataFrame(rows).sort_values("P_titre", ascending=False).reset_index(drop=True)

# Contrôle : exactement un champion par simulation
assert abs(res["P_titre"].sum() - 1.0) < 1e-9, "les P_titre ne somment pas a 1"

for col in ["P_titre", "P_finale", "P_demi", "P_quart", "P_8e", "P_qualifie"]:
    res[col] = (100 * res[col]).round(1)
res["buts_totaux_attendus"] = res["buts_totaux_attendus"].round(2)
res["matchs_joues_attendus"] = res["matchs_joues_attendus"].round(2)

res.to_csv("simulation_cdm2026.csv", index=False)
print("\n48 equipes enregistrees dans simulation_cdm2026.csv")
print("\nTop 15 (P_titre, P_finale, buts totaux attendus) :")
print(res.head(15)[["equipe", "P_titre", "P_finale", "P_demi",
                    "buts_totaux_attendus"]].to_string(index=False))
```

- [ ] **Step 2: Lancer la simulation**

Run: `.venv/bin/python simulate_wc2026.py`
Expected: imprime `Modele ajuste...`, `Distributions pre-calculees...`, `48 equipes enregistrees`, puis un top 15 avec des favoris (Argentine/Espagne/Brésil/France) en tête. Pas d'`AssertionError`.

- [ ] **Step 3: Vérifier le CSV**

Run: `.venv/bin/python -c "import pandas as pd; d=pd.read_csv('simulation_cdm2026.csv'); print(len(d), list(d.columns)); print(d.head())"`
Expected: 48 lignes, les 9 colonnes attendues.

- [ ] **Step 4: Commit** (git désactivé — sauter)

---

## Notes d'exécution

- **Git désactivé** : ignorer toutes les étapes « Commit ».
- **Toujours** `.venv/bin/python` / `.venv/bin/pytest`.
- **Task 1 a un checkpoint utilisateur** (Step 3) : montrer la correspondance groupes↔lettres et le squelette des 32es, attendre validation avant d'écrire le module.
- Si la simulation est lente (> ~2 min pour N=20000), réduire `N` à 10000 ; l'échantillonnage par `searchsorted` doit rester de l'ordre de quelques dizaines de secondes.
- Valeurs attendues : favoris (Argentine, Espagne, Brésil, France, Angleterre) avec P(titre) de l'ordre de 8-18 % ; somme des P_titre = 1.
