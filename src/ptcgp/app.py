"""Interface Streamlit, en local : ``python -m ptcgp app`` (ou ``streamlit run src/ptcgp/app.py``).

Deux modes :

- **Classé** : la vraie méta des tournois standard ; les extensions et promos cochées ne s'appliquent
  qu'à ton deck.
- **Règles pour tout le monde** : un format détecté dans les tournois CUSTOM et NOEX (Fusion Double
  seule, A1 à A3b…), dont les tournois donnent la méta. Retirer des extensions ou des cartes ne touche
  que ton deck.
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ptcgp import stats
from ptcgp.config import DATA_URL, PROCESSED, PROCESSED_FILES
from ptcgp.formats import STANDARD, main_sets
from ptcgp.http import HttpClient

# Couleur de la série (palette de référence, emplacement 1), une valeur par thème
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
MUTED = {"light": "#8c8b86", "dark": "#7d7c76"}
CLASSE, REGLES = "Classé", "Règles pour tout le monde"
CREDITS = (
    "Données de tournois : [Limitless TCG](https://play.limitlesstcg.com). Cartes : "
    "[flibustier/pokemon-tcg-pocket-database](https://github.com/flibustier/pokemon-tcg-pocket-database) (MIT), "
    "noms français : [TCGdex](https://tcgdex.dev). Projet non officiel, sans lien avec The Pokémon Company, "
    "Creatures, GAME FREAK ou DeNA."
)


# --------------------------------------------------------------------------- calculs mis en cache


def data_path() -> str:
    return os.environ.get("PTCGP_PROCESSED", str(PROCESSED))


def fetch_processed(url: str, dest: Path) -> Path:
    """Télécharge les tables publiées (branche « data ») dans ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    client = HttpClient(max_retries=3)
    for name in PROCESSED_FILES:
        (dest / name).write_bytes(client.get_bytes(f"{url}/{name}"))
    return dest


@st.cache_resource(ttl=24 * 3600, max_entries=1, show_spinner="Téléchargement des données…")
def remote_data(url: str) -> str:
    """Tables de la branche « data », rafraîchies au plus une fois par jour (app en ligne)."""
    return str(fetch_processed(url, Path(tempfile.mkdtemp(prefix="ptcgp-"))))


@st.cache_resource(max_entries=2, show_spinner="Chargement des données…")
def load_data(path: str) -> stats.Data:
    return stats.load(Path(path))


@st.cache_resource(max_entries=6, show_spinner="Calcul du classement…")
def run(
    path: str,
    format_id: str,
    since: str | None,
    sets: tuple[str, ...] | None,
    event_promos: bool,
    removed: tuple[str, ...],
    tolerance: int,
    min_lists: int,
) -> stats.Analysis:
    data = load_data(path)
    pool = None
    if format_id != STANDARD:
        pool = stats.rules_pool(data, format_id, sets=set(sets) if sets is not None else None, removed=set(removed))
    return stats.analyze(
        data,
        sets=set(sets) if sets is not None else None,
        event_promos=event_promos,
        since=since,
        max_missing=tolerance,
        min_lists=min_lists,
        format_id=format_id,
        pool=pool,
    )


# --------------------------------------------------------------------------- mise en forme


def theme() -> str:
    kind = getattr(getattr(st.context, "theme", None), "type", None)
    return kind if kind in SERIES else "light"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def set_label(sets: pd.DataFrame) -> dict[str, str]:
    names = sets["name_fr"].where(sets["name_fr"].fillna("") != "", sets["name"])
    return {s: f"{s} · {n}" for s, n in zip(sets["set"], names)}


def card_names(data: stats.Data) -> dict[str, str]:
    cards = data.cards
    french = cards["name_fr"].fillna(cards["name"]) if "name_fr" in cards else cards["name"]
    return dict(zip(cards["card_id"], french))


def ranking_chart(ranking: pd.DataFrame, top: int) -> alt.Chart:
    """Taux de victoire attendu de chaque deck, avec son intervalle à 90 % (une seule série)."""
    mode = theme()
    df = ranking.head(top).assign(
        win=lambda d: d["win_rate"] * 100,
        lo=lambda d: d["low"] * 100,
        hi=lambda d: d["high"] * 100,
        interval=lambda d: [f"{pct(r.low)} – {pct(r.high)}" for r in d.itertuples()],
        best=lambda d: d["p_best"].map(lambda x: pct(x, 0)),
        popularity=lambda d: d["share"].map(pct),
        label=lambda d: [f"{i}. {n}" for i, n in enumerate(d["name_fr"], 1)],
    )
    order = list(df["label"])
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    tooltip = [
        alt.Tooltip("name_fr:N", title="Deck"),
        alt.Tooltip("win:Q", title="Victoire attendue (%)", format=".1f"),
        alt.Tooltip("interval:N", title="Intervalle à 90 %"),
        alt.Tooltip("best:N", title="P(meilleur)"),
        alt.Tooltip("popularity:N", title="Popularité"),
    ]
    scale = alt.Scale(domain=[math.floor(min(df["lo"].min(), 50)) - 1, math.ceil(df["hi"].max()) + 1], nice=False, zero=False)
    axis = alt.Axis(tickMinStep=1, format="d")
    x = alt.X("win:Q", title="Victoire attendue par manche (%)", scale=scale, axis=axis)
    rules = alt.Chart(df).mark_rule(strokeWidth=2, color=SERIES[mode]).encode(x=alt.X("lo:Q", scale=scale, axis=axis), x2="hi:Q", y=y, tooltip=tooltip)
    points = alt.Chart(df).mark_circle(size=90, color=SERIES[mode], opacity=1).encode(x=x, y=y, tooltip=tooltip)
    # Cible de survol plus large que le point
    hits = alt.Chart(df).mark_bar(opacity=0, height=18).encode(x=alt.X("lo:Q", scale=scale, axis=axis), x2="hi:Q", y=y, tooltip=tooltip)
    half = alt.Chart(pd.DataFrame({"v": [50]})).mark_rule(strokeDash=[4, 4], strokeWidth=1, color=MUTED[mode]).encode(x="v:Q")
    return (half + rules + hits + points).properties(height=max(160, 28 * len(df)))


# --------------------------------------------------------------------------- page


def sidebar(data: stats.Data) -> dict:
    """Réglages ; renvoie les paramètres du calcul."""
    st.sidebar.header("Réglages")
    mode = st.sidebar.radio(
        "Mode",
        [CLASSE, REGLES],
        captions=["Vraie méta ; le filtre ne touche que ton deck", "Format spécial : mêmes règles pour tous"],
        disabled=data.formats.empty,
        help=None if not data.formats.empty else "Aucun format spécial détecté dans les données.",
    )
    labels = set_label(data.sets)
    first_day = date.fromisoformat(data.tournaments["date"].min()[:10])
    last_day = date.fromisoformat(data.tournaments["date"].max()[:10])
    params: dict = {"mode": mode}

    if mode == CLASSE:
        params["format_id"] = STANDARD
        since = min(max(date.fromisoformat(stats.default_since(data, STANDARD)), first_day), last_day)
        params["since"] = st.sidebar.date_input("Depuis le", since, min_value=first_day, max_value=last_day, format="DD/MM/YYYY", key="since_std").isoformat()
        main = list(main_sets(data.sets)["set"])
        every = st.sidebar.toggle("Toutes les extensions", value=True, help="Décoche pour ne garder que les extensions que tu possèdes.")
        if every:
            params["sets"] = None
        else:
            latest = stats.latest_expansion(data.sets)["set"]
            chosen = st.sidebar.multiselect("Extensions de ton deck", main, default=[latest], format_func=labels.get)
            params["sets"] = tuple(chosen)
        params["event_promos"] = st.sidebar.checkbox(
            "Promos d'événement",
            value=False,
            help="Les 7 dresseurs de boutique (Poké Ball, Recherches Professorales, Vitesse +…) sont toujours permis. "
            "Coche si tu as les promos gagnées lors d'événements passés.",
        )
        params["removed"] = ()
    else:
        formats = data.formats
        options = list(formats["format_id"])
        info = formats.set_index("format_id")
        fid = st.sidebar.selectbox(
            "Format",
            options,
            format_func=lambda f: f"{info.loc[f, 'label']} ({info.loc[f, 'tournaments']} tournoi{'s' if info.loc[f, 'tournaments'] > 1 else ''}, {info.loc[f, 'players']} joueurs)",
        )
        params["format_id"] = fid
        row = info.loc[fid]
        t = data.tournaments[stats.format_ids(data.tournaments) == fid]
        default = stats.default_since(data, fid) or t["date"].min()[:10]
        lo, hi = date.fromisoformat(t["date"].min()[:10]), date.fromisoformat(t["date"].max()[:10])
        params["since"] = st.sidebar.date_input(
            "Depuis le", max(lo, min(date.fromisoformat(default[:10]), hi)), min_value=lo, max_value=hi, format="DD/MM/YYYY", key=f"since_{fid}"
        ).isoformat()
        fmt_sets = row["sets"].split("|") if isinstance(row["sets"], str) and row["sets"] else list(main_sets(data.sets)["set"])
        chosen = st.sidebar.multiselect(
            "Extensions de ton deck", fmt_sets, default=fmt_sets, format_func=labels.get, key=f"sets_{fid}",
            help="Celles du format. En retirer une ne touche que ton deck : les adversaires gardent la méta du format.",
        )
        params["sets"] = tuple(chosen) if set(chosen) != set(fmt_sets) else None
        extra = data.format_cards[data.format_cards["format_id"] == fid]
        names = card_names(data)
        sets_of = dict(zip(data.cards["card_id"], data.cards["sets"]))
        if len(extra):
            kept = st.sidebar.multiselect(
                "Cartes permises en plus",
                list(extra["card_id"]),
                default=list(extra["card_id"]),
                format_func=lambda c: f"{names.get(c, c)} ({sets_of.get(c, '').replace('PROMO-', 'P-').replace('|', ', ')})",
                key=f"cards_{fid}",
                help="Cartes jouées dans ces tournois hors des extensions du format. Retire celles que tu n'as pas.",
            )
            params["removed"] = tuple(sorted(set(extra["card_id"]) - set(kept)))
        else:
            st.sidebar.caption("Pas d'autre carte permise : seuls les 7 dresseurs de boutique s'ajoutent aux extensions.")
            params["removed"] = ()
        params["event_promos"] = False
        rules = [r for r, on in (("aucun ex", row["no_ex"]), ("1 exemplaire par carte", row.get("singleton", False))) if on]
        if rules:
            st.sidebar.caption("Règle observée dans les listes : " + ", ".join(rules) + ".")

    params["tolerance"] = st.sidebar.slider(
        "Tolérance (exemplaires hors réserve par liste)", 0, 6, 0,
        help="Garde les listes auxquelles il manque quelques cartes, signalées « à trouver ou remplacer ». "
        "Le taux affiché reste celui de la liste d'origine.",
    )
    with st.sidebar.expander("Réglages avancés"):
        params["min_lists"] = st.slider("Listes jouables minimum par deck", 3, 30, 10)
        params["top"] = st.slider("Decks affichés dans le graphique", 5, 30, 15)
    return params


LIST_NOTE = (
    "Liste conseillée : celle qui a le meilleur taux de victoire estimé, à adversaire égal, parmi les listes "
    f"jouées à l'identique par au moins {stats.MIN_LIST_PLAYERS} joueurs. Sur des tournois que le modèle "
    "n'avait pas vus, les effets des cartes n'améliorent pas la prédiction : entre deux listes d'un même deck, "
    "l'écart reste du même ordre que le hasard."
)


def show_list(res: stats.Analysis, deck: str, signature: tuple, name: str, key: str) -> pd.DataFrame:
    """Les cartes d'une liste exacte (nombre, nom, version à chercher en jeu) et la liste à copier."""
    cards, _, _ = stats.deck_list(res.data, res.legal, deck, res.allowed, signature)
    if cards.empty:
        st.caption("Pas de liste disponible.")
        return cards
    st.dataframe(
        pd.DataFrame(
            {
                "Nb": cards["count"],
                "Carte": cards["name_fr"],
                "Version": cards["print"],
                "À trouver": cards["available"].map({True: "", False: "hors réserve"}),
            }
        ),
        hide_index=True,
        height=36 * len(cards) + 40,
        key=f"table_{key}",
    )
    if not cards["available"].all():
        st.caption("« Hors réserve » : carte absente de tes extensions, à trouver ou à remplacer (tolérance).")
    with st.expander(f"Copier la liste ({int(cards['count'].sum())} cartes)"):
        st.code(stats.list_text(name, cards), language=None)
    return cards


def explain_choice(res: stats.Analysis, deck: str, best: pd.Series, popular: pd.Series, cards: pd.DataFrame) -> None:
    """Pourquoi cette liste : joueurs, taux estimé, et écart avec la liste la plus jouée."""
    rate = f"taux estimé {pct(best['win_rate'])} [{pct(best['low'])} – {pct(best['high'])}]"
    if best["signature"] == popular["signature"]:
        st.caption(f"Jouée à l'identique par {best['players']} joueurs ({rate}). C'est aussi la liste la plus jouée.")
        return
    gap, low, high = stats.list_gap(res, deck, best["signature"], popular["signature"])
    other, _, _ = stats.deck_list(res.data, res.legal, deck, res.allowed, popular["signature"])
    names = {**dict(zip(other["card_id"], other["name_fr"])), **dict(zip(cards["card_id"], cards["name_fr"]))}
    prints = {**dict(zip(other["card_id"], other["print"])), **dict(zip(cards["card_id"], cards["print"]))}
    changes = ", ".join(f"{'+' if v > 0 else '−'}{abs(v)} {names.get(c, c)} ({prints.get(c, '')})"
                        for c, v in stats.list_difference(best["signature"], popular["signature"]).items())
    verdict = "écart net" if low > 0 else "écart incertain"
    pts = lambda x: f"{x * 100:+.1f}".replace(".", ",")
    st.caption(
        f"Jouée à l'identique par {best['players']} joueurs ({rate}). Par rapport à la liste la plus jouée "
        f"({popular['players']} joueurs) : {pts(gap)} pt [{pts(low)} ; {pts(high)}], {verdict}. "
        f"Cartes qui changent : {changes}."
    )


def recommended_deck(res: stats.Analysis) -> None:
    """En haut de page : le meilleur deck pour ces réglages et sa liste complète."""
    top = res.ranking.iloc[0]
    best, popular = res.recommended(top["deck"])
    st.subheader("Ton deck recommandé")
    st.markdown(f"#### {top['name_fr']}")
    cols = st.columns(3)
    cols[0].metric("Victoire attendue", pct(top["win_rate"]), help=f"Intervalle à 90 % : {pct(top['low'])} – {pct(top['high'])}")
    cols[1].metric("Chances d'être le meilleur", pct(top["p_best"], 0))
    cols[2].metric("Joueurs avec cette liste", f"{best['players']}" if best is not None else "–")
    if best is not None:
        cards = show_list(res, top["deck"], best["signature"], top["name_fr"], key="top")
        explain_choice(res, top["deck"], best, popular, cards)
    st.caption(LIST_NOTE)


def similar_formats(data: stats.Data, params: dict) -> list[str]:
    """Formats détectés dont les extensions sont toutes parmi celles cochées en mode classé."""
    if params["mode"] != CLASSE or not params["sets"] or data.formats.empty:
        return []
    chosen = set(params["sets"])
    fixed = data.formats[data.formats["sets"].fillna("") != ""]
    return [r.label for r in fixed.itertuples() if set(r.sets.split("|")) <= chosen]


def header(data: stats.Data, res: stats.Analysis, params: dict) -> None:
    window = res.data
    games = res.model.N.sum() / 2  # manches entre decks différents (miroirs exclus)
    cols = st.columns(4)
    cols[0].metric("Tournois", f"{len(window.tournaments)}")
    cols[1].metric("Joueurs", f"{len(window.entries):,}".replace(",", " "))
    cols[2].metric("Manches analysées", f"{games:,.0f}".replace(",", " "))
    cols[3].metric("Decks suivis", f"{len(res.model.decks) - 1}", help="Hors « Autres decks » (moins de 0,5 % des joueurs ou moins de 5 joueurs).")
    updated = date.fromisoformat(data.tournaments["date"].max()[:10])
    period = f"depuis le {date.fromisoformat(res.since[:10]):%d/%m/%Y}" if res.since else "sur tout l'historique"
    period += f" (tournois jusqu'au {updated:%d/%m/%Y}, mise à jour chaque semaine)"
    if params["mode"] == CLASSE:
        st.caption(f"Format standard, {period}. Le filtre ne s'applique qu'à ton deck : tes adversaires gardent la vraie méta.")
    else:
        row = data.formats.set_index("format_id").loc[params["format_id"]]
        st.caption(f"Format « {row['label']} », {period}. Ces règles valent pour tous : la méta est celle de ces tournois.")


def main() -> None:
    st.set_page_config(page_title="PTCGP Meta Analyzer", page_icon="🃏", layout="wide")
    path = data_path()
    if not (Path(path) / "tournaments.csv").exists():
        try:
            path = remote_data(os.environ.get("PTCGP_DATA_URL", DATA_URL))
        except Exception as err:  # hors ligne, ou branche « data » pas encore publiée
            st.error(f"Données indisponibles ({err}). En local, lance d'abord « python -m ptcgp update ».")
            st.stop()
    data = load_data(path)
    params = sidebar(data)

    st.title("PTCGP Meta Analyzer")
    if params["sets"] is not None and not params["sets"]:
        st.warning("Choisis au moins une extension.")
        st.stop()
    res = run(path, params["format_id"], params["since"], params["sets"], params["event_promos"], params["removed"], params["tolerance"], params["min_lists"])
    header(data, res, params)

    ranking = res.ranking
    if ranking.empty:
        st.warning("Aucun deck n'a assez de listes jouables avec ces réglages. Augmente la tolérance ou ajoute des extensions.")
        same = similar_formats(data, params)
        if same:
            st.info(f"Des tournois se jouent avec ces seules extensions : essaie le mode « {REGLES} », format « {same[0]} ».")
        st.caption(CREDITS)
        return

    recommended_deck(res)

    st.subheader("Classement")
    if len(ranking) < 3:
        st.info(
            f"Seulement {len(ranking)} deck{'s' if len(ranking) > 1 else ''} avec au moins {params['min_lists']} listes jouables : "
            "P(meilleur) a peu de sens. Baisse le seuil dans les réglages avancés pour en comparer davantage."
        )
    st.caption(
        "Victoire attendue par manche contre la méta, avec son intervalle à 90 %. P(meilleur) : part des 4 000 "
        "tirages où le deck finit premier. Aucun deck ne domine quand les intervalles se chevauchent."
    )
    table = pd.DataFrame(
        {
            "Deck": ranking["name_fr"],
            "Victoire attendue": ranking["win_rate"] * 100,
            "Intervalle à 90 %": [f"{pct(r.low)} – {pct(r.high)}" for r in ranking.itertuples()],
            "P(meilleur)": ranking["p_best"] * 100,
            "Popularité": ranking["share"] * 100,
            "Listes jouables": ranking["lists"],
            "Manches": ranking["games"].round().astype(int),
            "Pires matchups": ranking["worst"],
            "Nom Limitless": ranking["name"],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Victoire attendue": st.column_config.NumberColumn(format="%.1f %%"),
            "P(meilleur)": st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=100),
            "Popularité": st.column_config.NumberColumn(format="%.1f %%"),
            "Pires matchups": st.column_config.TextColumn(width="large"),
        },
    )
    st.altair_chart(ranking_chart(ranking, params["top"]), width="stretch")

    st.subheader("Détail d'un deck")
    deck = st.selectbox("Deck", list(ranking["deck"]), format_func=dict(zip(ranking["deck"], ranking["name_fr"])).get)
    row = ranking.set_index("deck").loc[deck]
    st.caption(f"Nom Limitless : {row['name']} · {row['lists']} listes jouables · {pct(row['win_rate'])} attendus [{pct(row['low'])} – {pct(row['high'])}]")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Matchups, du pire au meilleur**")
        m = stats.matchup_table(res, deck)
        st.dataframe(
            pd.DataFrame(
                {
                    "Adversaire": m["name_fr"],
                    "Victoire": m["win"] * 100,
                    "Intervalle à 90 %": [f"{pct(r.low, 0)} – {pct(r.high, 0)}" for r in m.itertuples()],
                    "Popularité": m["share"] * 100,
                    "Manches observées": m["games"].round().astype(int),
                }
            ),
            hide_index=True,
            column_config={
                "Victoire": st.column_config.NumberColumn(format="%.0f %%"),
                "Popularité": st.column_config.NumberColumn(format="%.1f %%"),
            },
            height=420,
        )
        st.caption("Peu de manches observées : le matchup est tiré vers la force globale des deux decks (Bradley-Terry).")
    with right:
        best, popular = res.recommended(deck)
        tab_best, tab_popular, tab_cards = st.tabs(["Liste conseillée", "La plus jouée", "Effet des cartes"])
        with tab_best:
            if best is not None:
                cards = show_list(res, deck, best["signature"], row["name_fr"], key=f"best_{deck}")
                explain_choice(res, deck, best, popular, cards)
        with tab_popular:
            if popular is not None:
                st.caption(f"{popular['players']} joueurs l'ont jouée à l'identique ; taux estimé {pct(popular['win_rate'])}.")
                show_list(res, deck, popular["signature"], row["name_fr"], key=f"popular_{deck}")
        with tab_cards:
            effects = stats.card_table(res, deck)
            if effects.empty:
                st.caption("Toutes les listes de ce deck jouent les mêmes cartes.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Carte": effects["name_fr"],
                            "Exemplaires (moyenne)": effects["mean_count"].round(1),
                            "Effet d'un exemplaire de plus": effects["effect"] * 100,
                            "Intervalle à 90 %": [f"{r.low * 100:+.1f} à {r.high * 100:+.1f}".replace(".", ",") for r in effects.itertuples()],
                        }
                    ),
                    hide_index=True,
                    column_config={"Effet d'un exemplaire de plus": st.column_config.NumberColumn(format="%+.1f pt")},
                )
                st.caption(
                    "Effet sur la victoire attendue, à adversaire égal, lissé vers 0 par Bayes empirique. "
                    "Les écarts dépassent rarement un point : le choix du deck compte bien plus que les retouches."
                )

    with st.expander("Tournois de la période"):
        t = res.data.tournaments.sort_values("date", ascending=False)
        st.dataframe(
            pd.DataFrame({"Date": t["date"].str[:10], "Tournoi": t["name"], "Joueurs": t["players"], "Decklists": t["decklists"]}),
            hide_index=True,
        )
    st.caption(CREDITS)


if __name__ == "__main__":  # « streamlit run » exécute le fichier sous ce nom
    main()
