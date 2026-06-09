"""
Données officielles du tableau de la Coupe du Monde 2026 (tirage du 05/12/2025).

GROUPS : lettre de groupe -> 4 équipes (noms tels qu'ils apparaissent dans le
         dataset martj42, ex. "United States", "South Korea").
R32    : 16 matchs des 32es dans l'ordre de l'arbre officiel. Chaque créneau est
         "1X"/"2X" (1er/2e du groupe X) ou "T1".."T8" (un des 8 meilleurs 3es).
         L'arbre au-delà est implicite : vainqueur(match 2k) vs vainqueur(2k+1),
         ce qui reproduit les 8es (M89-96), quarts (M97-100), demis (M101-102)
         et la finale (M104) officiels.

Sources : Wikipédia « 2026 FIFA World Cup draw » et « 2026 FIFA World Cup
          knockout stage ». Correspondance validée contre les 72 affiches du
          dataset (chaque groupe est bien une clique de matchs).
"""

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# 32es dans l'ordre de l'arbre : (match 2k, match 2k+1) s'affrontent en 8es.
R32 = [
    ("1E", "T1"),   # M74 : 1E vs 3e
    ("1I", "T2"),   # M77 : 1I vs 3e
    ("2A", "2B"),   # M73
    ("1F", "2C"),   # M75
    ("2K", "2L"),   # M83
    ("1H", "2J"),   # M84
    ("1D", "T3"),   # M81 : 1D vs 3e
    ("1G", "T4"),   # M82 : 1G vs 3e
    ("1C", "2F"),   # M76
    ("2E", "2I"),   # M78
    ("1A", "T5"),   # M79 : 1A vs 3e
    ("1L", "T6"),   # M80 : 1L vs 3e
    ("1J", "2H"),   # M86
    ("2D", "2G"),   # M88
    ("1B", "T7"),   # M85 : 1B vs 3e
    ("1K", "T8"),   # M87 : 1K vs 3e
]
