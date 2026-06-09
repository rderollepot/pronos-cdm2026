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

N = 500000                              # nombre de simulations
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
