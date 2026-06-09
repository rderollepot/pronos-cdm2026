"""
Logique pure de simulation du tournoi (sans I/O), pilotée par un RNG numpy.

Distribution de match : (cum, p1, p2) où cum = probas cumulées (121,) de la
matrice de score 11x11 aplatie (idx = i*11 + j ; i buts A, j buts B), p1 = P(A
gagne), p2 = P(B gagne). Voir bracket_2026 pour le schéma GROUPS / R32.

Niveaux atteints : 0 éliminé en poule · 1 qualifié (32es) · 2 en 8es · 3 en
quarts · 4 en demis · 5 en finale · 6 champion.
"""
from collections import defaultdict
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
