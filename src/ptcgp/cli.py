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


def _stats(args: argparse.Namespace) -> None:
    import pandas as pd

    from . import stats

    data = stats.load(PROCESSED)
    sets = {s.strip() for s in args.sets.split(",")} if args.sets else None
    if sets and sets - set(data.sets["set"]):
        unknown = ", ".join(sorted(sets - set(data.sets["set"])))
        raise SystemExit(f"Extensions inconnues : {unknown}. Codes valides : {', '.join(data.sets['set'])}")
    res = stats.analyze(data, sets=sets, include_promos=not args.no_promos, since=args.since, max_missing=args.tolerance)
    window = res.data
    since = args.since or stats.latest_expansion(data.sets)["release_date"]
    print(f"Période : depuis le {since} ; {len(window.tournaments)} tournois, {len(window.entries)} joueurs")
    print(f"Lissage des matchups : κ = {res.model.kappa:.0f} manches fictives ; {len(res.model.decks)} decks suivis (dont « Autres »)")
    if res.allowed is not None:
        tolerance = f" ; jusqu'à {args.tolerance} exemplaire(s) d'autres extensions toléré(s)" if args.tolerance else ""
        print(f"Extensions autorisées pour ton deck : {', '.join(sorted(res.allowed))}{tolerance}")
    if res.ranking.empty:
        print("Aucun deck n'a assez de listes jouables avec ces extensions.")
        return
    top = res.ranking.head(args.top)
    table = pd.DataFrame(
        {
            "Deck": top["name"],
            "Victoire attendue [90 %]": [f"{r.win_rate:.1%} [{r.low:.1%} ; {r.high:.1%}]" for r in top.itertuples()],
            "P(meilleur)": top["p_best"].map(lambda x: f"{x:.0%}"),
            "Popularité": top["share"].map(lambda x: f"{x:.1%}"),
            "Listes": top["lists"],
            "Pires matchups": top["worst"],
        }
    )
    print(table.to_string(index=False))

    best = res.ranking.iloc[0]
    cards, same, total = stats.deck_list(window, res.legal, best["deck"], res.allowed)
    print(f"\nListe la plus jouée de {best['name']} ({same} joueurs sur {total} listes jouables) :")
    for r in cards.itertuples():
        print(f"  {r.count}x {r.name:<30} {r.print}" + ("" if r.available else "   <- hors de tes extensions"))

    if args.validate:
        dates = window.tournaments["date"].sort_values().to_numpy()
        split = dates[int(len(dates) * 0.67)]
        scores = stats.holdout_scores(window, split)
        print(f"\nValidation : modèles appris avant le {split[:10]}, testés après ({scores.attrs['test_games']:.0f} manches)")
        print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


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
    p = sub.add_parser("stats", help="classement des decks pour le classé")
    p.add_argument("--sets", help="extensions autorisées pour son deck, séparées par des virgules (défaut : toutes)")
    p.add_argument("--no-promos", action="store_true", help="ne pas autoriser d'office les cartes promo")
    p.add_argument("--tolerance", type=int, default=0, help="exemplaires hors de ces extensions tolérés par liste (défaut : 0)")
    p.add_argument("--since", help="début de la période AAAA-MM-JJ (défaut : sortie de la dernière extension)")
    p.add_argument("--top", type=int, default=15, help="nombre de decks affichés")
    p.add_argument("--validate", action="store_true", help="tester les modèles sur les tournois les plus récents")
    args = parser.parse_args(argv)

    if args.command == "stats":
        _stats(args)
        return 0

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
