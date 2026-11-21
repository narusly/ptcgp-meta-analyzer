"""Construit les tables de ``data/processed`` à partir du cache brut (cartes et tournois)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import pandas as pd

from . import cards as cardsmod
from .limitless import TournamentCache, parse_pairings, parse_standings, records_from_pairings

TOURNAMENT_COLUMNS = ["tournament_id", "name", "date", "format", "players", "organizer_id", "decklists", "matches", "records_ok"]
ENTRY_COLUMNS = ["tournament_id", "player", "archetype_id", "archetype", "wins", "losses", "ties", "placing", "drop", "has_decklist"]
DECK_CARD_COLUMNS = ["tournament_id", "player", "card_id", "set", "number", "count"]
MATCH_COLUMNS = ["tournament_id", "phase", "round", "player1", "player2", "result", "games1", "games2", "archetype1", "archetype2"]


def build_cards(raw_cards: Path, out: Path) -> dict[str, pd.DataFrame]:
    data = cardsmod.load(raw_cards)
    sets = cardsmod.sets_table(data["sets.json"])
    prints = cardsmod.prints_table(data["cards.extra.json"])
    cards = cardsmod.cards_table(prints, sets)
    out.mkdir(parents=True, exist_ok=True)
    sets.to_csv(out / "sets.csv", index=False)
    prints.to_csv(out / "prints.csv", index=False)
    cards.to_csv(out / "cards.csv", index=False)
    if (raw_cards / "source.json").exists():
        shutil.copyfile(raw_cards / "source.json", out / "cards_source.json")
    return {"sets": sets, "prints": prints, "cards": cards}


def build_tournaments(cache: TournamentCache, prints: pd.DataFrame, out: Path) -> dict[str, pd.DataFrame]:
    metas, entries, deck_cards, matches = [], [], [], []
    for tid in cache.ids():
        meta = cache.load(tid, "meta")
        pairings = cache.load(tid, "pairings")
        t_entries, t_cards = parse_standings(tid, cache.load(tid, "standings"))
        t_matches = parse_pairings(tid, pairings)
        # Contrôle qualité : part des joueurs dont le bilan recalculé égale celui affiché par Limitless
        recomputed = records_from_pairings(pairings)
        same = sum(recomputed.get(e["player"], (0, 0, 0)) == (e["wins"], e["losses"], e["ties"]) for e in t_entries)
        metas.append(
            {
                "tournament_id": tid,
                "name": meta["name"],
                "date": meta["date"],
                "format": meta.get("format") or "STANDARD",
                "players": meta.get("players"),
                "organizer_id": meta.get("organizerId"),
                "decklists": sum(e["has_decklist"] for e in t_entries),
                "matches": len(t_matches),
                "records_ok": round(same / len(t_entries), 4) if t_entries else None,
            }
        )
        entries += t_entries
        deck_cards += t_cards
        matches += t_matches

    tournaments = pd.DataFrame(metas, columns=TOURNAMENT_COLUMNS).sort_values("date", ascending=False, ignore_index=True)
    entries_df = pd.DataFrame(entries, columns=ENTRY_COLUMNS).drop_duplicates(["tournament_id", "player"])

    cards_df = pd.DataFrame(deck_cards, columns=["tournament_id", "player", "set", "number", "count", "name"])
    cards_df["number"] = cards_df["number"].astype("Int64")
    keys = prints[["set", "number", "card_id"]].astype({"number": "Int64"})
    cards_df = cards_df.merge(keys, on=["set", "number"], how="left")
    unmapped = cards_df[cards_df["card_id"].isna()]

    matches_df = pd.DataFrame(matches, columns=MATCH_COLUMNS[:-2])
    archetypes = entries_df[["tournament_id", "player", "archetype_id"]]
    for side in ("1", "2"):
        matches_df = matches_df.merge(
            archetypes.rename(columns={"player": f"player{side}", "archetype_id": f"archetype{side}"}),
            on=["tournament_id", f"player{side}"],
            how="left",
        )

    names = (
        entries_df.dropna(subset=["archetype_id"])
        .groupby(["archetype_id", "archetype"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False).drop_duplicates("archetype_id")
        [["archetype_id", "archetype"]].sort_values("archetype_id", ignore_index=True)
    )

    out.mkdir(parents=True, exist_ok=True)
    tournaments.to_csv(out / "tournaments.csv", index=False)
    names.to_csv(out / "archetypes.csv", index=False)
    entries_df.to_csv(out / "entries.csv.gz", index=False)
    cards_df[DECK_CARD_COLUMNS].to_csv(out / "deck_cards.csv.gz", index=False)
    matches_df[MATCH_COLUMNS].to_csv(out / "matches.csv.gz", index=False)
    return {
        "tournaments": tournaments,
        "archetypes": names,
        "entries": entries_df,
        "deck_cards": cards_df,
        "matches": matches_df,
        "unmapped": unmapped,
    }


def summary(tables: dict[str, pd.DataFrame]) -> list[str]:
    t, e, m, u = tables["tournaments"], tables["entries"], tables["matches"], tables["unmapped"]
    lines = [
        f"Cartes : {len(tables['cards'])} cartes distinctes, {len(tables['prints'])} impressions, {len(tables['sets'])} extensions",
        f"Tournois : {len(t)}" + (f" (du {t['date'].min()[:10]} au {t['date'].max()[:10]})" if len(t) else ""),
        f"Joueurs inscrits : {len(e)}, dont {int(e['has_decklist'].sum())} avec decklist ; {e['archetype_id'].nunique()} archétypes",
        f"Matchs exploitables : {len(m)}",
    ]
    if len(t):
        weights = t["players"].fillna(0)
        share = (t["records_ok"].fillna(0) * weights).sum() / max(weights.sum(), 1)
        lines.append(f"Bilans recalculés identiques à Limitless : {share:.1%} des joueurs ; tournois à moins de 95 % : {int((t['records_ok'] < 0.95).sum())}")
    if len(u):
        missing = u.groupby(["set", "number", "name"], dropna=False).size().sort_values(ascending=False)
        lines.append(f"ATTENTION : {len(u)} lignes de decklist sans carte correspondante, par ex. {missing.head(5).to_dict()}")
    else:
        lines.append("Toutes les cartes des decklists sont retrouvées dans la base")
    return lines


def build_all(raw_cards: Path, cache: TournamentCache, out: Path, log: Callable[[str], None] = print) -> dict[str, pd.DataFrame]:
    tables = build_cards(raw_cards, out)
    tables.update(build_tournaments(cache, tables["prints"], out))
    for line in summary(tables):
        log(line)
    return tables
