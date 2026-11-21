from datetime import datetime, timezone

from ptcgp.limitless import TournamentCache, parse_pairings, parse_standings, records_from_pairings, select_tournaments


def test_records_from_pairings_follows_the_limitless_encoding():
    pairings = [
        {"phase": 1, "winner": "a", "player1": "a"},  # exemption gagnée
        {"phase": 1, "winner": -1, "player1": "d"},  # ligne seule notée -1 : défaite
        {"phase": 1, "winner": "b", "player1": "a", "player2": "b"},
        {"phase": 1, "winner": 0, "player1": "a", "player2": "c"},
        {"phase": 1, "winner": -1, "player1": "b", "player2": "c"},  # double défaite
        {"phase": 2, "winner": "c", "player1": "c", "player2": "d"},  # le top cut compte aussi
    ]
    assert records_from_pairings(pairings) == {"a": (1, 1, 1), "b": (1, 1, 0), "c": (1, 1, 1), "d": (0, 2, 0)}


def test_parse_pairings_results_and_skipped_rows():
    pairings = [
        {"round": 1, "phase": 1, "winner": -1, "games": {}, "player1": "solo"},  # exemption
        {"round": 1, "phase": 1, "winner": "b", "games": {"1": {"winner": "b"}}, "player1": "a", "player2": "b"},
        {"round": 2, "phase": 1, "winner": 0, "games": {}, "player1": "a", "player2": "c"},  # nul
        {"round": 3, "phase": 1, "winner": -1, "games": {}, "player1": "b", "player2": "c"},  # sans résultat
        {
            "round": 8, "phase": 2, "winner": "c", "player1": "c", "player2": "a",
            "games": {"1": {"winner": "c"}, "2": {"winner": "a"}, "3": {"winner": "c"}},
        },
    ]
    matches = parse_pairings("t1", pairings)
    assert [(m["player1"], m["player2"], m["result"]) for m in matches] == [("a", "b", 2), ("a", "c", 0), ("c", "a", 1)]
    assert (matches[-1]["games1"], matches[-1]["games2"]) == (2, 1)


def test_parse_standings_maps_sets_and_flags_missing_lists():
    standings = [
        {
            "player": "a",
            "deck": {"id": "mega-altaria-ex-b1-espeon-b3a", "name": "Mega Altaria ex Espeon"},
            "record": {"wins": 5, "losses": 2, "ties": 0},
            "placing": 3,
            "drop": None,
            "decklist": {
                "pokemon": [{"count": 2, "set": "B3a", "number": "20", "name": "Espeon"}],
                "trainer": [{"count": 2, "set": "P-A", "number": "7", "name": "Professor's Research"}],
            },
        },
        {"player": "b", "deck": None, "record": {"wins": 0, "losses": 3, "ties": 0}, "decklist": None},
    ]
    entries, cards = parse_standings("t1", standings)
    assert [e["has_decklist"] for e in entries] == [True, False]
    assert entries[1]["archetype_id"] is None
    assert [(c["set"], c["number"], c["count"]) for c in cards] == [("B3a", 20, 2), ("PROMO-A", 7, 2)]


def test_select_tournaments_keeps_finished_standard_events():
    now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
    since = datetime(2026, 10, 29, tzinfo=timezone.utc)
    tournaments = [
        {"id": "ok", "date": "2026-11-10T18:00:00.000Z", "format": None},
        {"id": "custom", "date": "2026-11-10T18:00:00.000Z", "format": "CUSTOM"},
        {"id": "noex", "date": "2026-11-10T18:00:00.000Z", "format": "NOEX"},
        {"id": "too-recent", "date": "2026-11-20T18:00:00.000Z", "format": None},
        {"id": "too-old", "date": "2026-10-01T18:00:00.000Z", "format": None},
    ]
    assert [t["id"] for t in select_tournaments(tournaments, since, now)] == ["ok"]


def test_cache_marks_a_tournament_complete_only_with_all_files(tmp_path):
    cache = TournamentCache(tmp_path)
    cache.save("t1", "standings", [])
    cache.save("t1", "pairings", [])
    assert not cache.has("t1") and cache.ids() == []
    cache.save("t1", "meta", {"id": "t1"})
    assert cache.has("t1") and cache.ids() == ["t1"]
    assert cache.load("t1", "meta") == {"id": "t1"}
