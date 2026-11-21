"""Ligne de commande : ``python -m ptcgp {cards,fetch,build,update}``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from . import cards
from .build import build_all
from .config import PROCESSED, RAW_CARDS, RAW_TOURNAMENTS
from .http import HttpClient
from .limitless import TournamentCache, fetch_tournaments, list_tournaments, select_tournaments

DEFAULT_DAYS = 90


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _fetch(since: datetime | None, max_new: int | None) -> None:
    now = datetime.now(timezone.utc)
    since = since or now - timedelta(days=DEFAULT_DAYS)
    client = HttpClient(min_interval=0.3)
    listed = list_tournaments(client, since)
    selected = select_tournaments(listed, since, now)
    print(f"{len(listed)} tournois Pocket depuis le {since:%Y-%m-%d}, {len(selected)} au format standard et terminés")
    fetch_tournaments(client, TournamentCache(RAW_TOURNAMENTS), selected, max_new=max_new)
    print(f"Requêtes envoyées à Limitless : {client.requests}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(prog="ptcgp", description="Données de méta Pokémon TCG Pocket (tournois Limitless et base de cartes)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cards", help="télécharger la base de cartes")
    for name, text in (("fetch", "télécharger les tournois Limitless absents du cache"), ("update", "cards, puis fetch, puis build")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--since", type=_date, help=f"date de début AAAA-MM-JJ (défaut : il y a {DEFAULT_DAYS} jours)")
        p.add_argument("--max", type=int, dest="max_new", help="nombre maximal de nouveaux tournois à télécharger")
    sub.add_parser("build", help="construire les tables de data/processed")
    args = parser.parse_args(argv)

    if args.command in ("cards", "update"):
        source = cards.download(HttpClient(), RAW_CARDS)
        print(f"Base de cartes téléchargée : {len(source['cards.extra.json'])} impressions")
    if args.command in ("fetch", "update"):
        _fetch(args.since, args.max_new)
    if args.command in ("build", "update"):
        build_all(RAW_CARDS, TournamentCache(RAW_TOURNAMENTS), PROCESSED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
