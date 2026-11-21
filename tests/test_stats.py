import numpy as np
import pandas as pd
import pytest

from ptcgp import stats

# --------------------------------------------------------------------------- jeu de données synthétique
# Trois decks en « pierre-feuille-ciseaux » : A bat B, B bat C, C bat A (75 % des manches).

BEATS = {("A", "B"), ("B", "C"), ("C", "A")}


def make_data(players_per_deck: int = 12, seed: int = 1) -> stats.Data:
    rng = np.random.default_rng(seed)
    sets = pd.DataFrame(
        {
            "set": ["PROMO-A", "B1", "C1"],
            "series": ["P", "B", "C"],
            "name": ["Promo A", "Mega Rising", "Dual Fusion"],
            "name_fr": ["", "", ""],
            "release_date": ["2024-10-30", "2025-10-30", "2026-10-29"],
            "count": [1, 1, 1],
        }
    )
    cards = pd.DataFrame(
        {
            "card_id": ["PK_1", "PK_2", "PK_3", "TR_7"],
            "name": ["Alpha", "Beta", "Gamma", "Professor's Research"],
            "category": ["pokemon", "pokemon", "pokemon", "supporter"],
            "sets": ["B1", "C1", "B1|C1", "PROMO-A"],
        }
    )
    prints = pd.DataFrame(
        {
            "card_id": ["PK_1", "PK_1", "PK_2", "PK_3", "PK_3", "TR_7"],
            "set": ["B1", "B1", "C1", "B1", "C1", "PROMO-A"],
            "number": [10, 250, 20, 30, 31, 7],
            "rarity": ["C", "SR", "C", "U", "C", "U"],
        }
    )
    deck_card = {"A": "PK_1", "B": "PK_2", "C": "PK_3"}
    entries, deck_cards = [], []
    for deck in "ABC":
        for k in range(players_per_deck):
            player = f"{deck}{k}"
            entries.append(
                {"tournament_id": "t1", "player": player, "archetype_id": deck.lower(), "archetype": f"Deck {deck}",
                 "wins": 0, "losses": 0, "ties": 0, "placing": None, "drop": None, "has_decklist": True,
                 "deck": deck, "deck_name": f"Deck {deck}"}
            )
            if deck == "A" and k == 0:  # 1 version normale + 1 version alternative
                deck_cards += [{"tournament_id": "t1", "player": player, "card_id": "PK_1", "set": "B1", "number": n, "count": 1} for n in (10, 250)]
            else:
                deck_cards.append({"tournament_id": "t1", "player": player, "card_id": deck_card[deck], "set": "B1", "number": 1, "count": 2})
            deck_cards.append({"tournament_id": "t1", "player": player, "card_id": "TR_7", "set": "PROMO-A", "number": 7, "count": 2})
    matches = []
    players = [(e["player"], e["deck"]) for e in entries]
    for i, (p1, d1) in enumerate(players):
        for p2, d2 in players[i + 1:]:
            if d1 == d2:
                continue
            p_win = 0.75 if (d1, d2) in BEATS else 0.25
            result = 1 if rng.random() < p_win else 2
            matches.append({"tournament_id": "t1", "phase": 1, "round": 1, "player1": p1, "player2": p2,
                            "result": result, "games1": int(result == 1), "games2": int(result == 2)})
    tournaments = pd.DataFrame({"tournament_id": ["t1"], "date": ["2026-11-01T10:00:00.000Z"], "records_ok": [1.0]})
    return stats.Data(sets, prints, cards, tournaments, pd.DataFrame(entries), pd.DataFrame(deck_cards), pd.DataFrame(matches))


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


# --------------------------------------------------------------------------- filtre par extension et listes


def test_legal_entries_with_and_without_tolerance():
    data = make_data()
    allowed = {"C1", "PROMO-A"}  # Alpha (deck A) n'existe qu'en B1
    strict = stats.legal_entries(data, allowed)
    by_deck = data.entries.assign(legal=strict).groupby("deck")["legal"].all()
    assert by_deck.to_dict() == {"A": False, "B": True, "C": True}
    assert stats.legal_entries(data, allowed, max_missing=2).all()
    assert stats.missing_copies(data, allowed).groupby(data.entries["deck"]).max().to_dict() == {"A": 2, "B": 0, "C": 0}


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
    assert list(table["category"])[0] == "pokemon"
