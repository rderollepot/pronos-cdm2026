"""Tests des primitives et de la simulation de tournoi sur entrées synthétiques."""
import numpy as np
import tournament as tn
import bracket_2026 as bk


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


def _toy_world():
    """Mini-univers déterministe : 12 groupes A-L de 4 équipes, où l'équipe est
    d'autant plus forte que son indice global est petit. Distributions dégénérées
    (la mieux classée gagne toujours 1-0)."""
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

    group_fixtures = {}
    for L, teams in groups.items():
        gf = []
        for i in range(4):
            for j in range(i + 1, 4):
                gf.append((teams[i], teams[j]))
        group_fixtures[L] = gf

    group_dist = {L: {m: dist(*m) for m in group_fixtures[L]} for L in groups}
    all_teams = [t for teams in groups.values() for t in teams]
    neutral_dist = {(a, b): dist(a, b) for a in all_teams for b in all_teams if a != b}
    return groups, group_fixtures, group_dist, neutral_dist


def test_build_bracket_yields_32_qualified():
    groups, gf, gd, nd = _toy_world()
    rng = np.random.default_rng(0)
    res = tn.simulate_tournament(rng, groups, gf, gd, nd, bk.R32)
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
    eliminated = [t for t, r in res.items() if r["level"] == 0]
    assert eliminated, "il doit rester des éliminés de poule"
    for t in eliminated:
        assert res[t]["matches"] == 3
