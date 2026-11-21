import pandas as pd

from ptcgp.build import build_tournaments, pseudonyms
from ptcgp.limitless import TournamentCache


def test_pseudonyms_follow_the_standings_and_cover_pairing_only_players():
    entries = [{"player": "alice"}, {"player": "bob"}]
    pairings = [{"player1": "bob", "player2": "carol"}, {"player1": "alice"}]
    assert pseudonyms(entries, pairings) == {"alice": "p0001", "bob": "p0002", "carol": "p0003"}


def tournament(cache: TournamentCache, tid: str, date: str) -> None:
    deck = {"id": "espeon-b3a", "name": "Espeon"}
    decklist = {"pokemon": [{"count": 2, "set": "B3a", "number": "20", "name": "Espeon"}], "trainer": []}
    cache.save(tid, "standings", [
        {"player": "alice", "deck": deck, "decklist": decklist, "record": {"wins": 1, "losses": 0, "ties": 0}, "placing": 1},
        {"player": "bob", "deck": deck, "decklist": decklist, "record": {"wins": 0, "losses": 1, "ties": 0}, "placing": 2},
    ])
    cache.save(tid, "pairings", [{"round": 1, "phase": 1, "winner": "alice", "games": {}, "player1": "alice", "player2": "bob"}])
    cache.save(tid, "meta", {"id": tid, "name": tid, "date": date, "format": None, "players": 2})


def test_published_tables_contain_no_player_names_and_drop_old_tournaments(tmp_path):
    cache = TournamentCache(tmp_path / "raw")
    tournament(cache, "recent", "2026-11-18T10:00:00.000Z")
    tournament(cache, "old", "2026-06-01T10:00:00.000Z")  # plus de 120 jours avant le plus récent
    prints = pd.DataFrame({"set": ["B3a"], "number": [20], "card_id": ["PK_018350"]})
    tables = build_tournaments(cache, prints, tmp_path / "out")
    assert list(tables["tournaments"]["tournament_id"]) == ["recent"]
    assert tables["tournaments"].loc[0, "records_ok"] == 1.0  # contrôle des bilans fait avant le remplacement
    for name, columns in (("entries", ["player"]), ("deck_cards", ["player"]), ("matches", ["player1", "player2"])):
        values = set(tables[name][columns].to_numpy().ravel())
        assert values <= {"p0001", "p0002"}, name
    for path in (tmp_path / "out").iterdir():
        text = pd.read_csv(path).to_csv()
        assert "alice" not in text and "bob" not in text
