import numpy as np
import pandas as pd
import pytest
from synthetic import BEATS, make_data, with_format

from ptcgp import stats
from ptcgp.formats import Pool

# --------------------------------------------------------------------------- modèles


def test_bradley_terry_recovers_known_strengths():
    true_pi = np.array([1.0, 2.0, 4.0])
    N = np.full((3, 3), 20000.0)
    np.fill_diagonal(N, 0)
    W = N * stats.bt_probabilities(true_pi)
    pi = stats.bradley_terry(W, N, prior_games=0.0)
    assert np.allclose(pi / pi[0], true_pi, rtol=1e-3)


def test_kappa_is_large_without_matchup_effects_and_small_with_them():
    rng = np.random.default_rng(0)
    pi = np.exp(rng.normal(0, 0.3, 12))
    p0 = stats.bt_probabilities(pi)
    N = np.full((12, 12), 200.0)
    np.fill_diagonal(N, 0)
    iu = np.triu_indices(12, 1)

    def simulate(p):
        W = np.zeros_like(N)
        W[iu] = rng.binomial(200, p[iu])
        W.T[iu] = 200 - W[iu]
        return W

    noisy = np.clip(p0 + rng.normal(0, 0.15, p0.shape), 0.05, 0.95)
    assert stats.fit_kappa(simulate(p0), N, p0) > 500
    assert stats.fit_kappa(simulate(noisy), N, p0) < 50


def test_fit_learns_the_rock_paper_scissors_matchups():
    model, _ = stats.fit(make_data(), min_share=0.0)
    p = pd.DataFrame(model.posterior_mean, index=model.decks, columns=model.decks)
    for a, b in BEATS:
        assert p.loc[a, b] > 0.6 and p.loc[b, a] < 0.4
    assert np.allclose(p.to_numpy() + p.to_numpy().T, 1.0)


def test_evaluate_without_filter_equals_the_full_model():
    data = make_data()
    model, _ = stats.fit(data, min_share=0.0)
    ranking = stats.evaluate(model, data, stats.legal_entries(data, None), min_lists=1, draws=500).set_index("deck")
    expected = model.posterior_mean @ model.meta["share"].to_numpy()
    for i, deck in enumerate(model.decks):
        assert ranking.loc[deck, "win_rate"] == pytest.approx(expected[i], abs=1e-6)
        assert ranking.loc[deck, "low"] <= ranking.loc[deck, "win_rate"] <= ranking.loc[deck, "high"]
    assert ranking["p_best"].sum() == pytest.approx(1.0)


def test_matchups_are_consistent_with_the_ranking():
    data = make_data()
    model, _ = stats.fit(data, min_share=0.0)
    ranking, m = stats.evaluate_matchups(model, data, stats.legal_entries(data, None), min_lists=1, draws=500)
    share = model.meta["share"].to_numpy()
    for r in ranking.itertuples():
        assert m.mean.loc[r.deck].to_numpy() @ share == pytest.approx(r.win_rate)
        assert (m.low.loc[r.deck] <= m.mean.loc[r.deck] + 1e-9).all() and (m.mean.loc[r.deck] <= m.high.loc[r.deck] + 1e-9).all()
    assert m.games.loc["A", "B"] == pytest.approx(144)  # 12 × 12 matchs d'une manche


def test_meta_table_needs_five_players_per_tracked_deck():
    entries = make_data(players_per_deck=4).entries
    meta = stats.meta_table(entries, min_share=0.0)
    assert list(meta.index) == [stats.OTHERS]  # 4 joueurs par deck : tous dans « Autres decks »


# --------------------------------------------------------------------------- réserve de son deck et listes


def test_legal_entries_with_and_without_tolerance():
    data = make_data()
    allowed = {"C1", "PROMO-A"}  # Alpha (deck A) n'existe qu'en B1, Delta (3 listes C) qu'en PROMO-B
    strict = stats.legal_entries(data, allowed)
    by_deck = data.entries.assign(legal=strict).groupby("deck")["legal"].all()
    assert by_deck.to_dict() == {"A": False, "B": True, "C": False}
    assert stats.legal_entries(data, allowed, max_missing=2).all()
    assert stats.missing_copies(data, allowed).groupby(data.entries["deck"]).max().to_dict() == {"A": 2, "B": 0, "C": 1}


def test_user_pool_keeps_shop_trainers_and_drops_event_promos_by_default():
    data = make_data()
    ok = stats.user_pool(data, {"C1"}).available(data.cards)
    assert ok["TR_7"] and ok["PK_2"]  # dresseur de boutique, carte de C1
    assert not ok["PK_4"] and not ok["PK_1"]  # promo d'événement, carte de B1
    assert stats.user_pool(data, {"C1"}, event_promos=True).available(data.cards)["PK_4"]
    assert not stats.user_pool(data).available(data.cards)["PK_4"]  # toutes les extensions, sans promos d'événement
    assert stats.user_pool(data, None, event_promos=True) is None  # aucune restriction


def test_filtered_candidates_exclude_unplayable_decks():
    data = make_data()
    model, _ = stats.fit(data, min_share=0.0)
    ranking = stats.evaluate(model, data, stats.legal_entries(data, {"C1", "PROMO-A"}), min_lists=1, draws=200)
    assert set(ranking["deck"]) == {"B", "C"}


def test_deck_list_merges_alternate_prints_and_flags_missing_cards():
    data = make_data()
    table, same, total = stats.deck_list(data, stats.legal_entries(data, None), "A", allowed={"C1", "PROMO-A"})
    rows = table.set_index("name")
    assert (same, total) == (12, 12)  # la liste à 1 + 1 versions est la même que les autres
    assert rows.loc["Alpha", "count"] == 2 and not rows.loc["Alpha", "available"]
    assert rows.loc["Alpha", "print"] == "B1-010"  # version la moins rare
    assert rows.loc["Professor's Research", "available"]
    assert rows.loc["Professor's Research", "name_fr"] == "Recherches Professorales"
    assert list(table["category"])[0] == "pokemon"


def test_deck_list_prefers_the_shop_version_of_shop_trainers():
    data = make_data()
    data.prints = pd.concat([data.prints, pd.DataFrame({"card_id": ["TR_7"], "set": ["C1"], "number": [99], "rarity": ["C"]})], ignore_index=True)
    data.cards.loc[data.cards["card_id"] == "TR_7", "sets"] = "PROMO-A|C1"
    pool = Pool(sets=frozenset({"C1"}), always=frozenset({"TR_7"}))
    table, _, _ = stats.deck_list(data, stats.legal_entries(data, None), "B", allowed=pool)
    assert table.set_index("name").loc["Professor's Research", "print"] == "P-A-007"


# --------------------------------------------------------------------------- formats


def test_select_keeps_only_the_requested_format():
    data = with_format(make_data())
    assert list(stats.select(data).tournaments["tournament_id"]) == ["t1"]
    assert list(stats.select(data, format_id="C1").tournaments["tournament_id"]) == ["t2"]
    assert stats.default_since(data, "C1") is None  # extensions fixes : tout l'historique


def test_rules_mode_ranks_the_format_and_filters_only_your_deck():
    data = with_format(make_data())
    res = stats.analyze(data, format_id="C1", min_share=0.0, min_lists=1, draws=200)
    assert set(res.ranking["deck"]) == {"B", "C"}  # Alpha (B1) n'est pas dans la réserve du format
    assert set(res.data.tournaments["tournament_id"]) == {"t2"}
    # Retirer la promo permise en plus : les 3 listes C qui la jouent ne comptent plus pour ton deck
    pool = stats.rules_pool(data, "C1", removed={"PK_4"})
    res2 = stats.analyze(data, format_id="C1", min_share=0.0, min_lists=1, draws=200, pool=pool)
    assert res2.ranking.set_index("deck").loc["C", "lists"] == 9
    assert set(res2.model.decks) == set(res.model.decks)  # la méta adverse ne change pas


def test_matchup_table_lists_opponents_from_worst_to_best():
    data = make_data()
    res = stats.analyze(data, sets=None, event_promos=True, min_share=0.0, min_lists=1, draws=200)
    table = stats.matchup_table(res, "A")
    assert list(table["opp"]) == ["C", "B"]  # A perd contre C, bat B
    assert table["win"].is_monotonic_increasing


def test_holdout_gaps_match_the_score_table():
    data = make_data(players_per_deck=12)
    late = data.matches.assign(tournament_id="t2")
    data.matches = pd.concat([data.matches, late], ignore_index=True)
    data.entries = pd.concat([data.entries, data.entries.assign(tournament_id="t2")], ignore_index=True)
    data.tournaments = pd.DataFrame({"tournament_id": ["t1", "t2"], "date": ["2026-11-01", "2026-11-05"], "records_ok": [1.0, 1.0]})
    result = stats.holdout_scores(data, "2026-11-03", min_share=0.0, bootstrap=200)
    scores = result.set_index("modèle")["log_loss"]
    gap, low, high = result.attrs["gaps"]["Pile ou face (50 %)"]
    assert gap == pytest.approx(scores["Bradley-Terry + lissage bayésien"] - scores["Pile ou face (50 %)"])
    assert low <= gap <= high
    assert gap < 0  # le modèle appris sur t1 prédit t2 mieux que le hasard


# --------------------------------------------------------------------------- listes conseillées


def test_card_model_finds_the_tech_card_and_recommends_its_list():
    data = make_data(players_per_deck=20, tech_boost=0.2)
    res = stats.analyze(data, event_promos=True, min_share=0.0, min_lists=1, draws=300)
    assert res.lists.attrs["tau"] > 0
    best, popular = res.recommended("B")
    assert ("TR_8", 1) in best["signature"]  # la liste avec « Tech Plus »
    effects = stats.card_table(res, "B").set_index("card_id")
    assert effects.loc["TR_8", "effect"] > 0 > effects.loc["TR_9", "effect"]
    plus = next(s for s in res.lists.loc[res.lists["deck"] == "B", "signature"] if ("TR_8", 1) in s)
    minus = next(s for s in res.lists.loc[res.lists["deck"] == "B", "signature"] if ("TR_9", 1) in s)
    gap, low, high = stats.list_gap(res, "B", plus, minus)
    assert low > 0 and low <= gap <= high
    cards, same, _ = stats.deck_list(res.data, res.legal, "B", None, best["signature"])
    assert same == 10 and cards["count"].sum() == 5  # 2 Bêta + 2 Recherches + 1 Tech Plus


def test_without_a_real_effect_lists_stay_close():
    res = stats.analyze(make_data(players_per_deck=20, tech_boost=1e-9), event_promos=True, min_share=0.0, min_lists=1, draws=300)
    lists = res.lists[res.lists["deck"] == "B"]
    assert (lists["win_rate"].max() - lists["win_rate"].min()) < 0.03


def test_recommended_list_needs_enough_players_or_falls_back_to_the_most_played():
    lists = pd.DataFrame({"deck": ["X"] * 3, "signature": [("a",), ("b",), ("c",)], "players": [40, 3, 6],
                          "win_rate": [0.50, 0.60, 0.52], "eligible": [True, False, True]})
    best, popular = stats.recommended_list(lists, "X")
    assert best["signature"] == ("c",) and popular["signature"] == ("a",)  # 3 joueurs : pas assez
    best, _ = stats.recommended_list(lists.assign(eligible=False), "X")
    assert best["signature"] == ("a",)


def test_list_text_lists_all_cards_with_versions():
    cards = pd.DataFrame({"count": [2, 1, 2], "name_fr": ["Mentali", "Tylton", "Poké Ball"], "category": ["pokemon", "pokemon", "item"],
                          "print": ["B3a-020", "B1-196", "P-A-005"], "available": [True, True, False]})
    text = stats.list_text("Deck", cards)
    assert text.startswith("Deck : 5 cartes") and "2× Mentali (B3a-020)" in text and "Dresseurs (2)" in text
    assert "← hors de ta réserve" in text
