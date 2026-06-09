# Pronostics Coupe du Monde 2026

Modèle statistique de prédiction de la Coupe du Monde 2026, du score d'un match
jusqu'à la probabilité de titre, à partir de l'historique des matchs internationaux.

## Le modèle

Modèle de Poisson sur forces d'équipes (**Maher 1982**), avec la correction
**Dixon-Coles (1997)** :

- le nombre de buts d'une équipe suit une loi de Poisson dont la moyenne dépend de
  sa force offensive, de la force défensive de l'adversaire et d'un avantage du
  terrain : `log(λ) = const + attaque[équipe] + défense[adversaire] + domicile` ;
- estimation par maximum de vraisemblance régularisé (Ridge), avec **pondération
  temporelle** (demi-vie 2 ans) et amicaux à poids réduit ;
- **correction Dixon-Coles** (`ρ`) sur les petits scores (0-0, 1-0, 0-1, 1-1),
  estimée en seconde étape ;
- **avantage hôte** appliqué aux 9 matchs de poule où USA / Mexique / Canada jouent
  chez eux (lu depuis la colonne `neutral` du dataset).

**Validation hors-échantillon** (matchs joués depuis 2024, jamais vus à
l'entraînement) : ~60 % d'exactitude sur le résultat 1/N/2, contre ~33 % au hasard.

**Données** : [martj42/international_results](https://github.com/martj42/international_results)
(49 000+ matchs internationaux 1872→2026, incluant le calendrier complet de la CDM
2026). Téléchargé automatiquement au premier lancement.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scipy pytest
```

## Utilisation

```bash
# Pronostics des 72 matchs de poule (1/N/2, score probable) -> pronostics_cdm2026.csv
.venv/bin/python predict_wc2026.py

# Validation : modèle de base (Poisson) vs Dixon-Coles, hors-échantillon
.venv/bin/python validate.py

# Simulation Monte-Carlo du tournoi complet -> simulation_cdm2026.csv
# (P(titre), P(finale/demi/quart/8e), buts totaux attendus par équipe)
.venv/bin/python simulate_wc2026.py

# Tests
.venv/bin/pytest
```

## Structure

| Fichier | Rôle |
|---|---|
| `wc_model.py` | Cœur du modèle : chargement, ajustement Poisson/Dixon-Coles, prédiction d'un match. |
| `predict_wc2026.py` | Pronostics des 72 matchs de poule → `pronostics_cdm2026.csv`. |
| `validate.py` | Validation hors-échantillon (base vs Dixon-Coles). |
| `bracket_2026.py` | Données officielles : 12 groupes (A-L) et squelette des 32es. |
| `tournament.py` | Logique de simulation d'un tournoi (poule, qualification, tableau, phase finale). |
| `simulate_wc2026.py` | Simulation Monte-Carlo (N tournois) → `simulation_cdm2026.csv`. |
| `tests/` | Tests unitaires (pytest). |
| `docs/superpowers/` | Specs de conception et plans d'implémentation. |

Paramètres réglables en tête de fichier : fenêtre d'entraînement, demi-vie, poids
des amicaux (`wc_model.py`) ; nombre de simulations `N` et graine (`simulate_wc2026.py`).

## Limites assumées

Modèle de **forme**, sans information d'effectif ni de blessures. La simulation
utilise un placement simplifié des 8 meilleurs troisièmes (pas la table FIFA
exacte des 495 combinaisons), sans confrontation directe dans les départages de
poule, sans avantage hôte en phase finale ni petite finale.
