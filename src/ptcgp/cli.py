"""Ligne de commande : ``python -m ptcgp {cards,fetch,build,update,stats,formats,app}``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    print(f"{len(listed)} tournois Pocket depuis le {since:%Y-%m-%d}, {len(selected)} terminés (standard, CUSTOM et NOEX)")
    fetch_tournaments(client, TournamentCache(RAW_TOURNAMENTS), selected, max_new=max_new)
    print(f"Requêtes envoyées à Limitless : {client.requests}")


def _formats() -> None:
    from . import stats

    data = stats.load(PROCESSED)
    if data.formats.empty:
        print("Aucun format à réserve restreinte détecté (lancer d'abord « python -m ptcgp build »).")
        return
    names = dict(zip(data.cards["card_id"], data.cards.get("name_fr", data.cards["name"]).fillna(data.cards["name"])))
    for f in data.formats.itertuples():
        extra = data.format_cards[data.format_cards["format_id"] == f.format_id]
        print(f"\n{f.label}  [--format {f.format_id}]")
        print(f"  {f.tournaments} tournoi(s), {f.players} joueurs, du {f.first_date[:10]} au {f.last_date[:10]} ; ex. {f.examples}")
        if len(extra):
            print("  Cartes permises en plus : " + ", ".join(f"{names.get(c, c)} ({k})" for c, k in zip(extra["card_id"], extra["lists"])))


def _stats(args: argparse.Namespace) -> None:
    import pandas as pd

    from . import stats

    data = stats.load(PROCESSED)
    sets = {s.strip() for s in args.sets.split(",")} if args.sets else None
    if sets and sets - set(data.sets["set"]):
        unknown = ", ".join(sorted(sets - set(data.sets["set"])))
        raise SystemExit(f"Extensions inconnues : {unknown}. Codes valides : {', '.join(data.sets['set'])}")
    if args.format != stats.STANDARD and (data.formats.empty or args.format not in set(data.formats["format_id"])):
        raise SystemExit(f"Format inconnu : {args.format}. Voir « python -m ptcgp formats ».")

    pool = None
    if args.format != stats.STANDARD and sets is not None:
        pool = stats.rules_pool(data, args.format, sets=sets)
    res = stats.analyze(
        data, sets=sets, event_promos=args.event_promos, since=args.since, max_missing=args.tolerance, format_id=args.format, pool=pool
    )
    window = res.data
    if args.format == stats.STANDARD:
        print(f"Format standard (classé) : le filtre ne s'applique qu'à ton deck, les adversaires gardent la vraie méta")
    else:
        label = data.formats.set_index("format_id").loc[args.format, "label"]
        print(f"Format « {label} » : règles pour tout le monde, méta de ses tournois")
    print(f"Période : {'depuis le ' + res.since if res.since else 'tout l’historique'} ; {len(window.tournaments)} tournois, {len(window.entries)} joueurs")
    print(f"Lissage des matchups : κ = {res.model.kappa:.0f} manches fictives ; {len(res.model.decks)} decks suivis (dont « Autres »)")
    if args.format == stats.STANDARD and sets is not None:
        tolerance = f" ; jusqu'à {args.tolerance} exemplaire(s) hors réserve toléré(s)" if args.tolerance else ""
        promos = "avec" if args.event_promos else "sans"
        print(f"Extensions de ton deck : {', '.join(sorted(sets))}, dresseurs de boutique, {promos} promos d'événement{tolerance}")
    elif args.format == stats.STANDARD and not args.event_promos:
        print("Ton deck : toutes les extensions et les dresseurs de boutique, sans promos d'événement (--event-promos pour les ajouter)")
    if res.ranking.empty:
        print("Aucun deck n'a assez de listes jouables avec ces règles.")
        return
    top = res.ranking.head(args.top)
    table = pd.DataFrame(
        {
            "Deck": top["name_fr"],
            "Victoire attendue [90 %]": [f"{r.win_rate:.1%} [{r.low:.1%} ; {r.high:.1%}]" for r in top.itertuples()],
            "P(meilleur)": top["p_best"].map(lambda x: f"{x:.0%}"),
            "Popularité": top["share"].map(lambda x: f"{x:.1%}"),
            "Listes": top["lists"],
            "Pires matchups": top["worst"],
        }
    )
    print(table.to_string(index=False))

    best = res.ranking.iloc[0]
    chosen, popular = res.recommended(best["deck"])
    cards_, same, total = stats.deck_list(window, res.legal, best["deck"], res.allowed, chosen["signature"])
    print(f"\nListe conseillée de {best['name_fr']} : meilleur taux estimé parmi les listes jouées par au moins "
          f"{stats.MIN_LIST_PLAYERS} joueurs ({same} joueurs, {chosen['win_rate']:.1%} [{chosen['low']:.1%} ; {chosen['high']:.1%}])")
    print(stats.list_text(best["name_fr"], cards_))
    if chosen["signature"] != popular["signature"]:
        gap, low, high = stats.list_gap(res, best["deck"], chosen["signature"], popular["signature"])
        print(f"Écart avec la liste la plus jouée ({popular['players']} joueurs) : {gap * 100:+.1f} pt [{low * 100:+.1f} ; {high * 100:+.1f}]")

    if args.validate:
        dates = window.tournaments["date"].sort_values().to_numpy()
        split = dates[int(len(dates) * 0.67)]
        scores = stats.holdout_scores(window, split)
        a = scores.attrs
        print(f"\nValidation : modèles appris avant le {split[:10]}, testés après ({a['test_tournaments']} tournois, {a['test_games']:.0f} manches)")
        print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("Écart de perte logarithmique par manche du modèle retenu [IC 95 % par bootstrap sur les tournois] :")
        for name, (gap, low, high) in a["gaps"].items():
            print(f"  contre {name:<22} {gap:+.4f} [{low:+.4f} ; {high:+.4f}]")
        h = stats.card_holdout(window, split)
        print(f"Effets des cartes (τ = {h['tau']:.3f} logit par exemplaire), mêmes tournois de test ({h['games']:.0f} manches) :")
        print(f"  perte logarithmique {h['base']:.4f} sans les cartes, {h['cards']:.4f} avec ; "
              f"écart {h['gap']:+.4f} [{h['low']:+.4f} ; {h['high']:+.4f}]")


def _app() -> int:
    """Lance l'interface Streamlit en local (http://localhost:8501), dans ce même processus."""
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py"))]
    return streamlit_cli.main()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(prog="ptcgp", description="Méta Pokémon TCG Pocket (tournois Limitless et base de cartes)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cards", help="télécharger la base de cartes et les noms français")
    for name, text in (("fetch", "télécharger les tournois Limitless absents du cache"), ("update", "cards, puis fetch, puis build")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--since", type=_date, help=f"date de début AAAA-MM-JJ (défaut : il y a {DEFAULT_DAYS} jours)")
        p.add_argument("--max", type=int, dest="max_new", help="nombre maximal de nouveaux tournois à télécharger")
    sub.add_parser("build", help="construire les tables de data/processed (et détecter les formats)")
    sub.add_parser("formats", help="formats à réserve restreinte détectés dans les tournois CUSTOM et NOEX")
    sub.add_parser("app", help="lancer l'interface Streamlit en local")
    sub.add_parser("figures", help="figure du README (reports/figures/classement.png)")
    p = sub.add_parser("stats", help="classement des decks")
    p.add_argument("--format", default="standard", help="format (défaut : standard, pour le classé ; voir la commande formats)")
    p.add_argument("--sets", help="extensions autorisées pour son deck, séparées par des virgules (défaut : toutes)")
    p.add_argument("--event-promos", action="store_true", help="autoriser les promos d'événement (les dresseurs de boutique le sont toujours)")
    p.add_argument("--tolerance", type=int, default=0, help="exemplaires hors réserve tolérés par liste (défaut : 0)")
    p.add_argument("--since", help="début de la période AAAA-MM-JJ (défaut : sortie de la dernière extension, ou tout l'historique d'un format à extensions fixes)")
    p.add_argument("--top", type=int, default=15, help="nombre de decks affichés")
    p.add_argument("--validate", action="store_true", help="tester les modèles sur les tournois les plus récents")
    args = parser.parse_args(argv)

    if args.command == "stats":
        _stats(args)
        return 0
    if args.command == "formats":
        _formats()
        return 0
    if args.command == "app":
        return _app()
    if args.command == "figures":
        from . import stats
        from .config import ROOT
        from .figures import ranking_figure

        print(ranking_figure(stats.load(PROCESSED), ROOT / "reports" / "figures" / "classement.png"))
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
