"""Jeu de données synthétique partagé par les tests des statistiques et de l'interface.

Trois decks en « pierre-feuille-ciseaux » : A bat B, B bat C, C bat A (75 % des manches).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ptcgp import stats

BEATS = {("A", "B"), ("B", "C"), ("C", "A")}


def make_data(players_per_deck: int = 12, seed: int = 1, tech_boost: float = 0.0) -> stats.Data:
    """``tech_boost`` > 0 : la moitié des joueurs du deck B jouent la carte « Tech Plus » au lieu de
    « Tech Moins », et gagnent ``tech_boost`` de plus contre tout adversaire."""
    rng = np.random.default_rng(seed)
    sets = pd.DataFrame(
        {
            "set": ["PROMO-A", "B1", "PROMO-B", "C1"],
            "series": ["A", "B", "B", "C"],
            "name": ["Promo A", "Mega Rising", "Promo B", "Dual Fusion"],
            "name_fr": ["Promo-A", "Méga-Ascension", "Promo-B", "Fusion Double"],
            "release_date": ["2024-10-30", "2025-10-30", "2025-10-30", "2026-10-29"],
            "count": [7, 1, 1, 1],
        }
    )
    cards = pd.DataFrame(
        {
            "card_id": ["PK_1", "PK_2", "PK_3", "PK_4", "TR_7", "TR_8", "TR_9"],
            "name": ["Alpha", "Beta", "Gamma", "Delta", "Professor's Research", "Tech Plus", "Tech Minus"],
            "name_fr": ["Alpha-fr", "Bêta", "Gamma-fr", "Delta-fr", "Recherches Professorales", "Tech Plus", "Tech Moins"],
            "category": ["pokemon", "pokemon", "pokemon", "pokemon", "supporter", "item", "item"],
            "sets": ["B1", "C1", "B1|C1", "PROMO-B", "PROMO-A", "C1", "C1"],
            "is_ex": [False, False, False, False, False, False, False],
        }
    )
    prints = pd.DataFrame(
        {
            "card_id": ["PK_1", "PK_1", "PK_2", "PK_3", "PK_3", "PK_4", "TR_7", "TR_8", "TR_9"],
            "set": ["B1", "B1", "C1", "B1", "C1", "PROMO-B", "PROMO-A", "C1", "C1"],
            "number": [10, 250, 20, 30, 31, 4, 7, 80, 90],
            "rarity": ["C", "SR", "C", "U", "C", "C", "U", "C", "C"],
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
                 "deck": deck, "deck_name": f"Deck {deck}", "deck_name_fr": f"Deck {deck} (fr)"}
            )
            if deck == "A" and k == 0:  # 1 version normale + 1 version alternative
                deck_cards += [{"tournament_id": "t1", "player": player, "card_id": "PK_1", "set": "B1", "number": n, "count": 1} for n in (10, 250)]
            else:
                deck_cards.append({"tournament_id": "t1", "player": player, "card_id": deck_card[deck], "set": "B1", "number": 1, "count": 2})
            if deck == "C" and k < 3:  # quelques listes jouent une promo d'événement
                deck_cards.append({"tournament_id": "t1", "player": player, "card_id": "PK_4", "set": "PROMO-B", "number": 4, "count": 1})
            deck_cards.append({"tournament_id": "t1", "player": player, "card_id": "TR_7", "set": "PROMO-A", "number": 7, "count": 2})
            if tech_boost and deck == "B":
                tech = "TR_8" if k % 2 == 0 else "TR_9"
                deck_cards.append({"tournament_id": "t1", "player": player, "card_id": tech, "set": "C1", "number": 80, "count": 1})
    plus = {f"B{k}" for k in range(0, players_per_deck, 2)} if tech_boost else set()
    matches = []
    players = [(e["player"], e["deck"]) for e in entries]
    for i, (p1, d1) in enumerate(players):
        for p2, d2 in players[i + 1:]:
            if d1 == d2:
                continue
            p_win = 0.75 if (d1, d2) in BEATS else 0.25
            p_win += tech_boost * ((p1 in plus) - (p2 in plus))
            result = 1 if rng.random() < p_win else 2
            matches.append({"tournament_id": "t1", "phase": 1, "round": 1, "player1": p1, "player2": p2,
                            "result": result, "games1": int(result == 1), "games2": int(result == 2)})
    tournaments = pd.DataFrame({"tournament_id": ["t1"], "date": ["2026-11-01T10:00:00.000Z"], "records_ok": [1.0]})
    return stats.Data(sets, prints, cards, tournaments, pd.DataFrame(entries), pd.DataFrame(deck_cards), pd.DataFrame(matches))


def with_format(data: stats.Data) -> stats.Data:
    """Ajoute un tournoi « t2 », copie de « t1 », classé dans un format « C1 seule »."""
    copy = lambda df: pd.concat([df, df.assign(tournament_id="t2")], ignore_index=True)
    tournaments = pd.DataFrame(
        {
            "tournament_id": ["t1", "t2"],
            "name": ["Standard Cup", "Fusion Only Cup"],
            "date": ["2026-11-01T10:00:00.000Z", "2026-11-02T10:00:00.000Z"],
            "format": ["STANDARD", "CUSTOM"],
            "players": [36, 36],
            "decklists": [36, 36],
            "records_ok": [1.0, 1.0],
            "format_id": ["standard", "C1"],
        }
    )
    formats = pd.DataFrame(
        {
            "format_id": ["C1"], "label": ["Fusion Double (C1) seule"], "sets": ["C1"], "no_ex": [False], "singleton": [False],
            "tournaments": [1], "players": [36], "lists": [36], "extra_cards": [1],
            "first_date": ["2026-11-02T10:00:00.000Z"], "last_date": ["2026-11-02T10:00:00.000Z"], "examples": ["Fusion Only Cup"],
        }
    )
    format_cards = pd.DataFrame({"format_id": ["C1"], "card_id": ["PK_4"], "lists": [3], "share": [3 / 36]})
    return stats.Data(
        data.sets, data.prints, data.cards, tournaments, copy(data.entries), copy(data.deck_cards), copy(data.matches),
        formats, format_cards,
    )


def write_processed(data: stats.Data, out: Path) -> Path:
    """Écrit les tables comme ``python -m ptcgp build`` (pour tester l'interface)."""
    out.mkdir(parents=True, exist_ok=True)
    data.sets.to_csv(out / "sets.csv", index=False)
    data.prints.to_csv(out / "prints.csv", index=False)
    data.cards.to_csv(out / "cards.csv", index=False)
    data.tournaments.to_csv(out / "tournaments.csv", index=False)
    data.entries.drop(columns=["deck", "deck_name", "deck_name_fr"]).to_csv(out / "entries.csv.gz", index=False)
    data.deck_cards.to_csv(out / "deck_cards.csv.gz", index=False)
    data.matches.to_csv(out / "matches.csv.gz", index=False)
    data.formats.to_csv(out / "formats.csv", index=False)
    data.format_cards.to_csv(out / "format_cards.csv", index=False)
    return out
