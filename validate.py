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
