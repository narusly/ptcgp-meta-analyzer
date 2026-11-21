"""Tournois Pokémon TCG Pocket de Limitless : liste, classements (decklists) et appariements.

API publique, sans clé : https://docs.limitlesstcg.com/developer. Chaque tournoi terminé est mis en
cache sur disque (``data/raw/limitless/<id>/``) et n'est plus jamais redemandé.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .cards import db_set_code
from .config import FETCHED_FORMATS, GAME, LIMITLESS_API, SETTLE_HOURS, STANDARD_FORMATS
from .http import HttpClient

PAGE_SIZE = 200
MAX_PAGES = 50


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def list_tournaments(client: HttpClient, since: datetime) -> list[dict]:
    """Tous les tournois Pocket (tous formats) commencés depuis ``since``, page par page."""
    found: dict[str, dict] = {}
    for page in range(1, MAX_PAGES + 1):
        batch = client.get_json(f"{LIMITLESS_API}/tournaments?game={GAME}&limit={PAGE_SIZE}&page={page}")
        recent = [t for t in batch if parse_date(t["date"]) >= since]
        found.update((t["id"], t) for t in recent)
        if len(batch) < PAGE_SIZE or not recent:
            break
    return sorted(found.values(), key=lambda t: t["date"], reverse=True)


def select_tournaments(
    tournaments: list[dict],
    since: datetime,
    now: datetime,
    formats: set | None = None,
    settle_hours: int = SETTLE_HOURS,
) -> list[dict]:
    """Tournois des ``formats`` voulus (défaut : standard, CUSTOM et NOEX), commencés depuis ``since``
    et terminés (au moins ``settle_hours`` h)."""
    formats = FETCHED_FORMATS if formats is None else formats
    cutoff = now - timedelta(hours=settle_hours)
    return [t for t in tournaments if t.get("format") in formats and since <= parse_date(t["date"]) <= cutoff]


def is_multideck(standings: list[dict], details: dict | None = None) -> bool:
    """Tournoi « Dual Deck » : chaque joueur inscrit deux decks (``decks`` et ``decklists`` au pluriel).

    Les appariements disent alors quel deck a joué chaque manche (``p1deck``, ``p2deck``), mais
    l'archétype d'un joueur n'est plus unique : ces tournois sont écartés des statistiques.
    """
    if details and details.get("multideck") is not None:
        return bool(details["multideck"])
    return any("decklists" in s or "decks" in s for s in standings)


class TournamentCache:
    """Un dossier par tournoi : ``meta.json`` (fiche de la liste), ``standings.json``, ``pairings.json``,
    et ``details.json`` pour les formats non standard (cartes bannies, tournoi Dual Deck)."""

    KINDS = ("meta", "standings", "pairings")

    def __init__(self, root: Path):
        self.root = root

    def path(self, tournament_id: str, kind: str) -> Path:
        return self.root / tournament_id / f"{kind}.json"

    def has(self, tournament_id: str) -> bool:
        return all(self.path(tournament_id, kind).exists() for kind in self.KINDS)

    def save(self, tournament_id: str, kind: str, data) -> None:
        path = self.path(tournament_id, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def load(self, tournament_id: str, kind: str):
        return json.loads(self.path(tournament_id, kind).read_text(encoding="utf-8"))

    def load_optional(self, tournament_id: str, kind: str):
        path = self.path(tournament_id, kind)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and self.has(p.name))


def fetch_tournaments(
    client: HttpClient,
    cache: TournamentCache,
    tournaments: list[dict],
    max_new: int | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Télécharge classements et appariements des tournois absents du cache ; renvoie le nombre ajouté.

    Pour les formats non standard, la fiche ``details`` (cartes bannies, Dual Deck) est aussi gardée.
    """
    todo = [t for t in tournaments if not cache.has(t["id"])]
    cached = len(tournaments) - len(todo)
    if max_new is not None:
        todo = todo[:max_new]
    log(f"{len(tournaments)} tournois retenus, {cached} déjà en cache, {len(todo)} à télécharger")
    for i, t in enumerate(todo, 1):
        tid = t["id"]
        standings = client.get_json(f"{LIMITLESS_API}/tournaments/{tid}/standings")
        pairings = client.get_json(f"{LIMITLESS_API}/tournaments/{tid}/pairings")
        cache.save(tid, "standings", standings)
        cache.save(tid, "pairings", pairings)
        if t.get("format") not in STANDARD_FORMATS:
            cache.save(tid, "details", client.get_json(f"{LIMITLESS_API}/tournaments/{tid}/details"))
        cache.save(tid, "meta", t)  # écrit en dernier : sa présence signifie « tournoi complet »
        fmt = t.get("format") or "standard"
        log(f"[{i}/{len(todo)}] {t['date'][:10]}  {fmt:<8} {t.get('players', '?'):>4} joueurs  {t['name'][:60]}")
    return len(todo)


def parse_standings(tournament_id: str, standings: list[dict]) -> tuple[list[dict], list[dict]]:
    """Une ligne par joueur (archétype, bilan) et une ligne par carte de sa decklist."""
    entries, cards = [], []
    for s in standings:
        deck = s.get("deck") or {}
        record = s.get("record") or {}
        decklist = s.get("decklist")
        has_list = isinstance(decklist, dict) and any(decklist.get(k) for k in ("pokemon", "trainer"))
        entries.append(
            {
                "tournament_id": tournament_id,
                "player": s["player"],
                "archetype_id": deck.get("id"),
                "archetype": deck.get("name"),
                "wins": record.get("wins", 0),
                "losses": record.get("losses", 0),
                "ties": record.get("ties", 0),
                "placing": s.get("placing"),
                "drop": s.get("drop"),
                "has_decklist": has_list,
            }
        )
        if not has_list:
            continue
        for section in ("pokemon", "trainer"):
            for c in decklist.get(section) or []:
                number = str(c.get("number", ""))
                cards.append(
                    {
                        "tournament_id": tournament_id,
                        "player": s["player"],
                        "set": db_set_code(c.get("set", "")),
                        "number": int(number) if number.isdigit() else None,
                        "count": int(c.get("count", 1)),
                        "name": c.get("name"),
                    }
                )
    return entries, cards


def records_from_pairings(pairings: list[dict]) -> dict[str, tuple[int, int, int]]:
    """Bilan (victoires, défaites, nuls) de chaque joueur, recalculé à partir des appariements.

    Codage de ``winner``, vérifié sur les bilans affichés par Limitless (phase suisse et top cut) :
    nom d'un joueur = son match gagné ; 0 = nul ; -1 entre deux joueurs = double défaite. Une ligne
    sans adversaire est une victoire par exemption si le joueur est noté vainqueur, une défaite sinon.
    """
    records: dict[str, list[int]] = {}

    def add(player: str, slot: int) -> None:
        records.setdefault(player, [0, 0, 0])[slot] += 1

    for p in pairings:
        p1, p2, winner = p.get("player1"), p.get("player2"), p.get("winner")
        if not p1:
            continue
        if not p2:
            add(p1, 0 if winner == p1 else 1)
        elif winner == p1:
            add(p1, 0), add(p2, 1)
        elif winner == p2:
            add(p2, 0), add(p1, 1)
        elif winner == 0 and not isinstance(winner, bool):
            add(p1, 2), add(p2, 2)
        else:
            add(p1, 1), add(p2, 1)
    return {player: tuple(r) for player, r in records.items()}


def parse_pairings(tournament_id: str, pairings: list[dict]) -> list[dict]:
    """Un match par ligne : ``result`` vaut 1 ou 2 (vainqueur), 0 pour un nul.

    Écartés, car ils ne disent rien sur quel deck bat lequel : les lignes sans adversaire (exemptions)
    et les doubles défaites (``winner`` = -1). Voir ``records_from_pairings`` pour le codage complet.
    """
    matches = []
    for p in pairings:
        p1, p2 = p.get("player1"), p.get("player2")
        if not p1 or not p2:
            continue
        winner = p.get("winner")
        if winner == p1:
            result = 1
        elif winner == p2:
            result = 2
        elif winner == 0 and not isinstance(winner, bool):
            result = 0
        else:
            continue
        games1 = games2 = 0
        for game in (p.get("games") or {}).values():
            if game.get("winner") == p1:
                games1 += 1
            elif game.get("winner") == p2:
                games2 += 1
        matches.append(
            {
                "tournament_id": tournament_id,
                "phase": p.get("phase"),
                "round": p.get("round"),
                "player1": p1,
                "player2": p2,
                "result": result,
                "games1": games1,
                "games2": games2,
            }
        )
    return matches
