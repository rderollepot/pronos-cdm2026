# Simulation Monte-Carlo de la CDM 2026 — Design

Date : 2026-06-09
Auteur : Romain Derollepot (avec Claude)

## Objectif

Simuler l'intégralité de la Coupe du Monde 2026 (phase de poule + phase à
élimination directe) par méthode de Monte-Carlo, en réutilisant le modèle de
Poisson Dixon-Coles existant, pour estimer :

1. **P(titre)** : la probabilité de victoire finale de chacune des 48 équipes.
2. **Buts totaux attendus** marqués par chaque équipe sur toute la compétition.

Bonus quasi gratuit (sous-produit de la simulation) : P(atteindre les 8es /
quarts / demis / finale) et P(qualifié en phase finale).

## Contexte

État de départ : modèle modulaire `wc_model.py` (Poisson Maher / Dixon-Coles,
avantage hôte, validé hors-échantillon). Le dataset `martj42/international_results`
contient les 72 matchs de poule (scores `NA`), mais **pas** les matchs de la phase
finale (non encore tirés, car dépendants des résultats de poule).

Constat de faisabilité : les **12 groupes** (4 équipes / 6 matchs chacun) se
reconstruisent parfaitement à partir des 72 affiches (composantes connexes du
graphe « joue contre en poule »). Il manque seulement les **lettres officielles**
des groupes (A-L) et le **squelette du tableau final**, à récupérer par recherche
web et figer dans `bracket_2026.py`.

Niveau de fidélité retenu (décision utilisateur) : **option B** — squelette
officiel exact des 32es pour les 24 premiers/deuxièmes de groupe ; placement
**simplifié et documenté** des 8 meilleurs troisièmes (la table FIFA exacte des 495
combinaisons est hors-scope, son impact sur les probas de titre est négligeable).

## Architecture

| Fichier | Responsabilité |
|---|---|
| `tournament.py` (nouveau) | Logique pure du tournoi : `rank_group`, `select_best_thirds`, `build_bracket`, `simulate_match`, `simulate_tournament`. Sans I/O, testable sur entrées synthétiques. |
| `simulate_wc2026.py` (nouveau) | Driver : ajuste le modèle, reconstruit les groupes, pré-calcule les matrices de score, lance N simulations, agrège, écrit le CSV. |
| `bracket_2026.py` (nouveau) | Données figées : lettres officielles des groupes (A-L) et squelette officiel des 32es (positions de groupe → créneaux), récupérés par recherche web puis validés. |
| `tests/test_tournament.py` (nouveau) | Tests unitaires de la logique de tournoi sur entrées synthétiques. |

`wc_model.py`, `predict_wc2026.py`, `validate.py` restent inchangés.

## Moteur de simulation (un tirage)

### Pré-calcul (une fois, avant les N simulations)
- Ajustement du modèle sur tout l'historique (`estimate_rho=True`, comme
  `predict_wc2026.py`).
- Matrice de score corrigée Dixon-Coles pré-calculée pour **tous les couples
  ordonnés** d'équipes en terrain neutre (48×48), plus les 9 matchs hôtes de poule
  avec avantage. Chaque échantillonnage de match devient un simple tirage
  catégoriel dans la matrice en cache (flatten + `rng.choice` sur les indices),
  ce qui rend 20 000 simulations rapides (quelques secondes).

### Échantillonnage d'un match — `simulate_match(rng, key_a, key_b, matrices, knockout)`
- Tire un score `(buts_a, buts_b)` depuis la matrice en cache du couple.
- En phase de poule : renvoie le score tel quel.
- En phase finale (`knockout=True`) : si `buts_a == buts_b`, on désigne le
  vainqueur par une séance de tirs au but modélisée par un Bernoulli de probabilité
  `p1 / (p1 + p2)` (forces relatives, où `p1`/`p2` sont les probas de victoire
  hors nul issues de la matrice). Aucun but supplémentaire n'est ajouté.

### Phase de poule
- 72 matchs simulés.
- Classement par **points (3/1/0) → différence de buts → buts marqués → tirage au
  sort aléatoire**. Simplification assumée : pas de confrontation directe.

### Qualification
- Les 2 premiers de chaque groupe (24 équipes).
- Les **8 meilleurs troisièmes** parmi les 12, classés points → diff → buts marqués
  → aléatoire.

### Tableau final
- Squelette officiel des 32es (`bracket_2026.py`) pour les 24 premiers/deuxièmes.
- Les 8 troisièmes qualifiés placés dans les 8 créneaux « 3e » par une règle
  déterministe documentée, évitant qu'un troisième retombe d'entrée sur une équipe
  de son propre groupe.
- Rounds : 32es (16 matchs) → 16es (8) → quarts (4) → demis (2) → finale (1).
  Pas de petite finale (hors-scope ; n'affecte ni le titre ni — significativement —
  les buts).
- Phase finale en terrain neutre (pas d'avantage hôte).

### Buts totaux
Pour chaque simulation, on additionne les buts tirés par chaque équipe sur **tous
ses matchs joués** (poule + phase finale, score du temps réglementaire). Moyenné sur
N simulations → buts totaux attendus par équipe. Les tirs au but n'ajoutent pas de
but.

## Agrégation et sortie

Sur N simulations, on compte par équipe : nombre de titres, de finales, de demis,
de quarts, de 8es, de qualifications en phase finale ; somme des buts marqués ; somme
des matchs joués. On divise par N.

Sortie `simulation_cdm2026.csv`, une ligne par équipe (48), colonnes :
`equipe, P_titre, P_finale, P_demi, P_quart, P_8e, P_qualifie,
buts_totaux_attendus, matchs_joues_attendus`. Triée par `P_titre` décroissant.
Top 15 imprimé en console.

## Tests (`tests/test_tournament.py`, données synthétiques)

- `rank_group` classe correctement par points, puis diff. de buts, puis buts marqués.
- `select_best_thirds` retient les 8 meilleurs troisièmes selon le même ordre.
- Un tournoi déterministe (matrices dégénérées où une équipe gagne toujours) donne
  `P(titre) = 1` pour cette équipe.
- Les buts totaux d'une équipe éliminée en poule = somme sur ses 3 matchs de groupe.

## Hors-scope (volontaire)

- Table FIFA exacte des 495 combinaisons de meilleurs troisièmes.
- Confrontation directe dans les départages de groupe.
- Avantage hôte en phase finale.
- Petite finale (3e place).
- Blessures / effectifs.

## Critères de réussite

- `simulate_wc2026.py` produit `simulation_cdm2026.csv` (48 équipes) avec P(titre),
  les probas tour-par-tour, les buts totaux attendus, et imprime un top 15 crédible
  (favoris en tête, cohérents avec le classement des forces du modèle).
- La somme des `P_titre` sur les 48 équipes vaut 1 (± tolérance Monte-Carlo).
- `pytest tests/test_tournament.py -v` passe.
- N (nombre de simulations) paramétrable en tête de `simulate_wc2026.py`
  (défaut 20 000).

## Dépendance externe

Recherche web en début d'implémentation pour figer dans `bracket_2026.py` :
- la correspondance groupes reconstruits ↔ lettres officielles A-L,
- le squelette officiel des 32es (quelle position de groupe occupe quel créneau).

La correspondance sera montrée à l'utilisateur pour validation avant utilisation.
