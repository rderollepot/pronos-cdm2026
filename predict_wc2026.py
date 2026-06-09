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
