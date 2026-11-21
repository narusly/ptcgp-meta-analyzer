import math

import numpy as np
import pandas as pd
import pytest

from ptcgp import formats
from ptcgp.formats import Pool

# --------------------------------------------------------------------------- monde synthétique
# Extensions : A1 ; B1, B2, B3 et la compilation B4b (réimpressions de B1) ; C1. Six cartes par
# extension, dont une ex en C1. Les listes standard jouent deux cartes de chaque extension.

SETS = pd.DataFrame(
    {
        "set": ["A1", "PROMO-A", "B1", "B2", "B3", "B4b", "C1"],
        "series": ["A", "A", "B", "B", "B", "B", "C"],
        "name": ["Genetic Apex", "Promo A", "Mega Rising", "Fantastical Parade", "Pulsing Aura", "Deluxe Pack: Mega", "Dual Fusion"],
        "name_fr": ["Puissance Génétique", "Promo-A", "Méga-Ascension", "Parade Onirique", "Aura Palpitante", "Booster de Luxe Méga", "Fusion Double"],
        "release_date": ["2024-10-30", "2024-10-30", "2025-10-30", "2026-01-29", "2026-04-28", "2026-09-30", "2026-10-29"],
    }
)
POOL = {s: [f"{s}_{i}" for i in range(6)] for s in ["A1", "B1", "B2", "B3", "C1"]}
SHOP = "TR_shop"
PROMO = "PK_promo"  # promo d'événement, uniquement en PROMO-A


def cards_table() -> pd.DataFrame:
    rows = []
    for s, ids in POOL.items():
        for c in ids:
            sets = s + ("|B4b" if s == "B1" else "")
            rows.append({"card_id": c, "name": c, "sets": sets, "first_set": s, "is_ex": c == "C1_0"})
    rows += [
        {"card_id": SHOP, "name": "Poké Ball", "sets": "PROMO-A", "first_set": "PROMO-A", "is_ex": False},
        {"card_id": PROMO, "name": "Event", "sets": "PROMO-A", "first_set": "PROMO-A", "is_ex": False},
    ]
    return pd.DataFrame(rows)


def lists(tid: str, n: int, groups: list[list[str]], rng, per_group: int = 2, extra: list[str] = (), extra_every: int = 0, count: int = 2):
    rows = []
    for k in range(n):
        player = f"{tid}_p{k}"
        chosen = [c for g in groups for c in rng.choice(g, size=min(per_group, len(g)), replace=False)]
        if extra_every and k % extra_every == 0:
            chosen += list(extra)
        for c in chosen + [SHOP]:
            rows.append({"tournament_id": tid, "player": player, "card_id": c, "count": count})
    return rows


def world(customs: dict[str, dict], seed: int = 0):
    """Tournois standard de référence + tournois CUSTOM décrits par ``customs``."""
    rng = np.random.default_rng(seed)
    everything = list(POOL.values())
    dc, t = [], []
    for i in range(4):  # standard : 4 × 50 listes, toutes extensions
        tid = f"std{i}"
        dc += lists(tid, 50, everything, rng)
        t.append({"tournament_id": tid, "name": tid, "date": f"2026-11-1{i}T10:00:00.000Z", "format": "STANDARD", "players": 50, "records_ok": 1.0, "multideck": False})
    for tid, spec in customs.items():
        dc += lists(tid, spec["n"], spec["groups"], rng, extra=spec.get("extra", []), extra_every=spec.get("extra_every", 0), count=spec.get("count", 2))
        t.append({"tournament_id": tid, "name": spec.get("name", tid), "date": "2026-11-15T10:00:00.000Z", "format": "CUSTOM",
                  "players": spec["n"], "records_ok": 1.0, "multideck": spec.get("multideck", False)})
    dc = pd.DataFrame(dc)
    entries = dc[["tournament_id", "player"]].drop_duplicates().assign(has_decklist=True)
    return pd.DataFrame(t), entries, dc


def detect(customs):
    t, entries, dc = world(customs)
    cols, table, extra = formats.detect(t, entries, dc, cards_table(), SETS, {SHOP})
    return t.merge(cols, on="tournament_id").set_index("tournament_id"), table.set_index("format_id"), extra


C1_ONLY = [POOL["C1"]]
NO_EX = [POOL["A1"], POOL["B1"], POOL["B2"], POOL["B3"], POOL["C1"][1:]]


# --------------------------------------------------------------------------- tests


def test_binom_cdf_matches_the_direct_sum():
    n, p = 30, 0.2
    for k in (0, 3, 10, 29):
        direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
        assert formats.binom_cdf(k, n, p) == pytest.approx(direct)
    assert formats.binom_cdf(30, 30, 0.5) == 1.0


def test_single_set_format_with_permitted_promo_and_small_tournament_joining_it():
    t, table, extra = detect({
        "fusion": {"n": 60, "groups": C1_ONLY, "extra": [PROMO], "extra_every": 4, "name": "Dual Fusion Only"},
        "fusion_small": {"n": 15, "groups": C1_ONLY},
    })
    assert t.loc["fusion", "format_id"] == t.loc["fusion_small", "format_id"] == "C1"
    assert table.loc["C1", "label"] == "Fusion Double (C1) seule"
    assert table.loc["C1", "tournaments"] == 2
    assert set(extra.loc[extra["format_id"] == "C1", "card_id"]) == {PROMO}  # la boutique est toujours permise
    assert t.loc["std0", "format_id"] == formats.STANDARD


def test_series_format_includes_the_compilation_it_implies():
    t, table, _ = detect({"bonly": {"n": 60, "groups": [POOL["B1"], POOL["B2"], POOL["B3"]]}})
    fid = t.loc["bonly", "format_id"]
    assert table.loc[fid, "sets"] == "B1|B2|B3|B4b"  # B4b ne contient que des cartes de B1
    assert table.loc[fid, "label"] == "Série B (B1 à B4b)"


def test_no_ex_and_singleton_rules():
    t, table, _ = detect({
        "noex": {"n": 60, "groups": NO_EX},
        "glc": {"n": 60, "groups": NO_EX, "count": 1},
    })
    assert t.loc["noex", "format_id"] == "toutes-sans-ex"
    assert t.loc["glc", "format_id"] == "toutes-sans-ex-1-exemplaire"
    assert table.loc["toutes-sans-ex", "label"] == "Toutes les extensions · sans ex"
    assert pd.isna(table.loc["toutes-sans-ex", "sets"]) or table.loc["toutes-sans-ex", "sets"] == ""


def test_full_pool_custom_and_hole_in_a_series_are_not_formats():
    t, table, _ = detect({
        "mystery": {"n": 60, "groups": list(POOL.values())},  # règle invisible dans les listes
        "hole": {"n": 60, "groups": [POOL["A1"], POOL["B1"], POOL["B3"], POOL["C1"]]},  # B2 absente entre B1 et B3
        "dual": {"n": 60, "groups": C1_ONLY, "multideck": True},
    })
    assert t[["format_id"]].loc[["mystery", "hole", "dual"]].isna().all().all()
    assert len(table) == 0


def test_pool_available_handles_sets_extra_cards_removals_and_ex():
    cards = cards_table()
    pool = Pool(sets=frozenset({"C1"}), cards=frozenset({PROMO}), always=frozenset({SHOP}))
    ok = pool.available(cards)
    assert ok["C1_3"] and ok[PROMO] and ok[SHOP] and not ok["B1_0"]
    assert not Pool(sets=frozenset({"C1"}), removed=frozenset({"C1_3"})).available(cards)["C1_3"]
    no_ex = Pool(no_ex=True).available(cards)
    assert not no_ex["C1_0"] and no_ex["C1_1"] and no_ex["B1_0"]
