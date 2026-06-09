# Dixon-Coles + avantage hôte + validation — Design

Date : 2026-06-09
Auteur : Romain Derollepot (avec Claude)

## Objectif

Améliorer le modèle de pronostics de la Coupe du Monde 2026 sur trois points :

1. **Correction Dixon-Coles** : corriger l'hypothèse d'indépendance des deux scores
   pour les petits scores (0-0, 1-0, 0-1, 1-1), là où le modèle de Poisson de base
   est mal calibré (notamment sur les nuls).
2. **Avantage hôte** : appliquer un avantage du terrain aux 9 matchs de poule où un
   pays hôte (USA, Mexique, Canada) joue chez lui.
3. **Validation avant/après** : mesurer objectivement si Dixon-Coles améliore les
   prédictions, par une évaluation hors-échantillon comparant base vs Dixon-Coles.

## Contexte

État de départ : un script monolithique `predict_wc2026.py` (modèle de Poisson
Maher / Dixon-Coles partiel, pondération temporelle, pénalité Ridge) qui produit
`pronostics_cdm2026.csv` (72 matchs de poule). Dataset : `martj42/international_results`.

Deux constats issus de l'inspection des données :

- L'avantage hôte est **déjà encodé** dans le dataset : les colonnes `country` et
  `neutral` marquent `neutral=FALSE` pour exactement les 9 matchs où un hôte joue
  chez lui (3 hôtes × 3 matchs de poule). Le pays hôte y est listé comme `home_team`.
  Le souci : la boucle de prédiction actuelle force `neutre=True` pour tous les
  matchs et ignore donc cette information.
- Les scores de la CDM 2026 sont la chaîne littérale `NA` (interprétée comme NaN
  par pandas) → les 72 fixtures restent bien identifiables comme « à prédire ».

## Architecture

Refactor léger justifié par le besoin de validation : la validation doit réutiliser
**exactement** la même logique d'ajustement que les pronostics, sinon la comparaison
n'a pas de sens. On extrait donc le modèle dans un module partagé.

| Fichier | Rôle |
|---|---|
| `wc_model.py` (nouveau) | Modèle réutilisable : chargement, pondération, ajustement Poisson, estimation de ρ, prédiction d'un match corrigée Dixon-Coles. |
| `predict_wc2026.py` (réécrit, mince) | Ajuste sur tout l'historique, prédit les 72 fixtures avec leur vrai `neutral`, écrit le CSV. |
| `validate.py` (nouveau) | Cache une période récente, ajuste base vs Dixon-Coles sur le même train, sort un tableau comparatif. |

### Interface du module `wc_model.py`

- `load_data(path) -> DataFrame` : charge le CSV (téléchargement auto si absent).
- `make_train(df, window_start, date_ref) -> DataFrame` : matchs joués dans la fenêtre.
- `fit_model(train, *, lambda_ridge, half_life, friendly_weight, date_ref,
  estimate_rho=True) -> FittedModel` : ajuste attaque/défense/domicile (inchangé) puis,
  si `estimate_rho`, estime ρ ; sinon ρ=0 (modèle de base).
- `FittedModel.predict(team_a, team_b, neutral=True) -> dict` : renvoie
  `lam_a, lam_b, p1, pN, p2, score` avec correction Dixon-Coles appliquée.

Le modèle de base et le modèle Dixon-Coles sont le **même** code avec `estimate_rho`
False/True : c'est ce qui garantit une comparaison juste en validation.

## 1. Correction Dixon-Coles (estimation en deux étapes)

Approche en deux étapes (ρ quasi indépendant des autres paramètres, cf.
Dixon & Coles 1997) : on garde l'ajustement attaque/défense/domicile actuel, puis on
estime ρ par une optimisation 1-D.

Pour chaque match d'entraînement, on calcule λ (buts attendus domicile) et μ (buts
attendus extérieur). On maximise la log-vraisemblance Dixon-Coles pondérée ; comme
seul τ dépend de ρ, cela revient à maximiser `Σ wₘ · log τ(xₘ, yₘ ; λₘ, μₘ, ρ)`.
Seuls les matchs à petit score (les deux scores ≤ 1) contribuent.

Facteur de correction (x = buts domicile, y = buts extérieur) :

```
τ(0,0) = 1 − λ·μ·ρ
τ(1,0) = 1 + μ·ρ
τ(0,1) = 1 + λ·ρ
τ(1,1) = 1 − ρ
τ(x,y) = 1   sinon
```

- Optimisation : `scipy.optimize.minimize_scalar` (Brent borné), ρ dans une plage
  sûre garantissant τ > 0 sur tous les matchs contributifs.
- **En prédiction** : matrice de scores `M = outer(P(domicile=i), P(extérieur=j))`,
  on multiplie les 4 cases (0,0),(1,0),(0,1),(1,1) par leur τ, puis on **renormalise**
  `M /= M.sum()` avant de calculer P(1)/P(N)/P(2) et le score le plus probable.

## 2. Avantage hôte

Aucune liste codée en dur. La boucle de prédiction lit le vrai `neutral` de chaque
fixture :

- `neutral=FALSE` (9 matchs hôtes) → l'équipe hôte (= `home_team`) reçoit le
  coefficient « domicile » déjà estimé par le modèle.
- `neutral=TRUE` (63 matchs) → terrain neutre, pas d'avantage.

Effet attendu : Mexique, USA, Canada gagnent un bonus de buts (≈ ×1,29) sur leurs
matchs à domicile. À vérifier : que le pays hôte est bien `home_team` dans les 9
lignes concernées (sinon appliquer l'avantage au camp marqué hôte).

## 3. Validation (avant/après)

- **Hold-out** : matchs réellement joués depuis une date de coupure récente
  (`TEST_START`, p.ex. 2024-01-01). Le train = matchs joués avant `TEST_START` dans
  la fenêtre habituelle.
- On ajuste **deux modèles sur le même train** : base (`estimate_rho=False`) et
  Dixon-Coles (`estimate_rho=True`). Les deux utilisent le `neutral` réel des matchs
  test → la comparaison isole proprement l'apport de Dixon-Coles.
- **Métriques** :
  - Exactitude 1/N/2 (argmax = résultat réel).
  - Log-loss sur les probas 1/N/2 (plus bas = mieux).
  - Calibration des nuls : P(nul) prédite moyenne vs fréquence réelle des nuls.
- **Sortie** : tableau comparatif imprimé `base vs Dixon-Coles`.

Note : le fix avantage hôte n'affecte pas le train (seulement les 72 fixtures), il se
vérifie donc à part en montrant que les 9 matchs hôtes voient leurs probas changer
dans le CSV régénéré.

## Hors-scope (volontaire)

- Simulation Monte-Carlo de la phase à élimination directe.
- Blessures / effectifs.
- RPS (on s'en tient à exactitude + log-loss).

## Critères de réussite

- `validate.py` imprime un tableau comparant base et Dixon-Coles sur exactitude,
  log-loss et calibration des nuls, sur une période hors-échantillon.
- `predict_wc2026.py` régénère `pronostics_cdm2026.csv` avec Dixon-Coles, et les 9
  matchs hôtes montrent un avantage à domicile (buts attendus du hôte revus à la
  hausse vs version neutre).
- `wc_model.py` est partagé par les deux scripts ; aucune duplication de la logique
  d'ajustement.
