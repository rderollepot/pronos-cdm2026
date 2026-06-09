"""Cohérence des données du tableau 2026 avec les fixtures du dataset."""
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
