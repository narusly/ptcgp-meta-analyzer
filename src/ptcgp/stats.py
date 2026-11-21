"""Statistiques de méta : popularité, matchups lissés et taux de victoire attendu.

Méthode (détaillée dans le README) :

1. L'unité d'observation est la manche, car le classé se joue en une manche. Un nul vaut une
   demi-victoire. Les decks trop rares (moins de ``min_share`` des joueurs, et moins de 5 joueurs)
   sont regroupés en « Autres decks ».
2. Force globale de chaque deck : modèle de Bradley-Terry, P(A bat B) = πA / (πA + πB).
3. Matchups : le résultat observé de A contre B est lissé vers la prédiction de Bradley-Terry, avec
   un poids κ (en manches fictives) estimé par Bayes empirique sur la vraisemblance bêta-binomiale.
   A posteriori, P(A bat B) suit une loi Bêta(victoires + κ·p0, défaites + κ·(1 − p0)).
4. Taux de victoire attendu : moyenne des matchups pondérée par la popularité des decks adverses
   (miroir compté à 50 %).
5. Incertitude : tirages de Monte-Carlo (lois bêta des matchups, loi de Dirichlet de la méta),
   d'où un intervalle à 90 % et la probabilité d'être le meilleur deck.

Deux modes :

- **Classé** (format standard) : le filtre (extensions, promos) ne s'applique qu'à son propre deck.
  Seules ses listes jouables comptent pour le deck candidat ; ses adversaires gardent la vraie méta,
  et ses autres listes servent d'a priori (voir ``evaluate_matchups``).
- **Règles pour tout le monde** : on se place dans un format détecté (voir formats.py), dont les
  tournois donnent la méta. Retirer des cartes de la réserve du format ne touche que son deck.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .archetypes import add_decks
from .cards import shop_card_ids
from .formats import STANDARD, Pool, as_pool, format_pool

OTHERS = "autres"
OTHERS_NAME = "Autres decks"
MAIN_SET = re.compile(r"^[A-Z]\d+[a-z]?$")
RARITY_ORDER = ["C", "U", "R", "RR", "AR", "SR", "SAR", "IM", "S", "SSR", "UR", "AN"]
MIN_PLAYERS = 5  # un deck suivi à part doit compter au moins 5 joueurs


# --------------------------------------------------------------------------- données


@dataclass
class Data:
    sets: pd.DataFrame
    prints: pd.DataFrame
    cards: pd.DataFrame
    tournaments: pd.DataFrame
    entries: pd.DataFrame
    deck_cards: pd.DataFrame
    matches: pd.DataFrame
    formats: pd.DataFrame = field(default_factory=pd.DataFrame)
    format_cards: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["format_id", "card_id", "lists", "share"]))


def load(processed: Path) -> Data:
    cards = pd.read_csv(processed / "cards.csv")
    entries = pd.read_csv(processed / "entries.csv.gz")
    pokemon = cards[cards["category"] == "pokemon"]
    french = dict(zip(pokemon["name"], pokemon["name_fr"])) if "name_fr" in cards else None
    optional = {name: processed / f"{name}.csv" for name in ("formats", "format_cards")}
    return Data(
        sets=pd.read_csv(processed / "sets.csv"),
        prints=pd.read_csv(processed / "prints.csv"),
        cards=cards,
        tournaments=pd.read_csv(processed / "tournaments.csv"),
        entries=add_decks(entries, pokemon["name"], french),
        deck_cards=pd.read_csv(processed / "deck_cards.csv.gz"),
        matches=pd.read_csv(processed / "matches.csv.gz"),
        **{name: pd.read_csv(path) for name, path in optional.items() if path.exists()},
    )


def latest_expansion(sets: pd.DataFrame) -> pd.Series:
    """Dernière extension principale sortie (les promos et packs spéciaux sont ignorés)."""
    main = sets[sets["set"].str.match(MAIN_SET)]
    return main.sort_values("release_date").iloc[-1]


def format_ids(tournaments: pd.DataFrame) -> pd.Series:
    """Format de chaque tournoi ; sans colonne ``format_id`` (anciennes tables), tout est standard."""
    if "format_id" in tournaments:
        return tournaments["format_id"]
    return pd.Series(STANDARD, index=tournaments.index)


MIN_RECENT_PLAYERS = 300


def default_since(data: Data, format_id: str = STANDARD) -> str | None:
    """Début de période par défaut : la sortie de la dernière extension pour le standard et les formats
    « toutes extensions » (sans ex…), qui changent à chaque extension ; tout l'historique pour un format
    à extensions fixes (C1 seule, A1 à A3b…), ou pour un format « toutes extensions » qui compte moins
    de ``MIN_RECENT_PLAYERS`` joueurs depuis la dernière extension."""
    release = latest_expansion(data.sets)["release_date"]
    if format_id == STANDARD:
        return release
    row = data.formats[data.formats["format_id"] == format_id] if len(data.formats) else pd.DataFrame()
    fixed_sets = len(row) and isinstance(row.iloc[0]["sets"], str) and row.iloc[0]["sets"]
    if fixed_sets:
        return None
    t = data.tournaments[format_ids(data.tournaments) == format_id]
    return release if t.loc[t["date"] >= release, "players"].fillna(0).sum() >= MIN_RECENT_PLAYERS else None


def select(data: Data, since: str | None = None, min_records_ok: float = 0.95, format_id: str = STANDARD) -> Data:
    """Tournois d'un format (par défaut le standard), depuis ``since``, au résultat fiable.

    ``since=None`` : début par défaut du format (voir ``default_since``). ``min_records_ok`` écarte
    les tournois dont les bilans recalculés ne collent pas, comme les tournois par équipes que l'API
    classe en format standard. Les tournois Dual Deck n'ont pas de format (voir formats.detect).
    """
    since = since or default_since(data, format_id)
    t = data.tournaments
    kept = t[(format_ids(t) == format_id) & (t["records_ok"] >= min_records_ok)]
    if since:
        kept = kept[kept["date"] >= since]
    ids = set(kept["tournament_id"])
    return replace(
        data,
        tournaments=kept,
        entries=data.entries[data.entries["tournament_id"].isin(ids)],
        deck_cards=data.deck_cards[data.deck_cards["tournament_id"].isin(ids)],
        matches=data.matches[data.matches["tournament_id"].isin(ids)],
    )


# --------------------------------------------------------------------------- réserve de son deck


def promo_sets(sets: pd.DataFrame) -> set[str]:
    return {s for s in sets["set"] if s.startswith("PROMO")}


def user_pool(data: Data, sets: set[str] | None = None, event_promos: bool = False) -> Pool | None:
    """Réserve de son deck en classé : extensions cochées (défaut : toutes), dresseurs de boutique
    toujours, promos d'événement seulement si ``event_promos``. None : aucune restriction."""
    if sets is None and event_promos:
        return None
    main = set(data.sets["set"]) - promo_sets(data.sets) if sets is None else set(sets)
    allowed = main | (promo_sets(data.sets) if event_promos else set())
    return Pool(sets=frozenset(allowed), always=frozenset(shop_card_ids(data.prints)))


def rules_pool(data: Data, format_id: str, sets: set[str] | None = None, removed: set[str] | None = None) -> Pool:
    """Réserve de son deck dans un format : celle du format, moins les extensions et cartes retirées."""
    row = data.formats.set_index("format_id").loc[format_id].copy()
    row["format_id"] = format_id
    pool = format_pool(row, data.format_cards, shop_card_ids(data.prints))
    if sets is not None:
        base = pool.sets if pool.sets is not None else frozenset(data.sets["set"])
        pool = replace(pool, sets=frozenset(base & set(sets)))
    return replace(pool, removed=frozenset(removed or ()))


def available_cards(cards: pd.DataFrame, allowed) -> pd.Series:
    """True pour les cartes jouables avec ``allowed`` (réserve, ensemble d'extensions ou None)."""
    pool = as_pool(allowed)
    if pool is None:
        return pd.Series(True, index=cards["card_id"])
    return pool.available(cards)


def missing_copies(data: Data, allowed) -> pd.Series:
    """Nombre d'exemplaires de la decklist introuvables dans la réserve (0 si tout est jouable)."""
    ok = available_cards(data.cards, allowed)
    dc = data.deck_cards
    missing = dc["count"].where(~dc["card_id"].map(ok).fillna(False).astype(bool), 0)
    per_entry = missing.groupby([dc["tournament_id"], dc["player"]]).sum()
    keys = pd.MultiIndex.from_frame(data.entries[["tournament_id", "player"]])
    return pd.Series(per_entry.reindex(keys).fillna(0).to_numpy(), index=data.entries.index)


def legal_entries(data: Data, allowed, max_missing: int = 0) -> pd.Series:
    """True pour les joueurs dont la decklist est jouable avec la réserve ``allowed``.

    ``max_missing`` tolère quelques exemplaires hors réserve (à trouver ou à remplacer) : les decks de
    méta mélangent beaucoup d'extensions, et un filtre strict sur une ou deux extensions ne laisse
    souvent aucun deck. ``allowed=None`` : toutes les cartes.
    """
    has_list = data.entries["has_decklist"].astype(bool)
    if as_pool(allowed) is None:
        return has_list
    return has_list & (missing_copies(data, allowed) <= max_missing)


# --------------------------------------------------------------------------- méta et manches


def versions(deck: str) -> str:
    """Extensions des Pokémon d'un deck, d'après sa clé : « vivillon@b2+butterfree@a3 » → « A3, B2 »."""
    parts = [p.rsplit("@", 1)[1].upper() for p in deck.split("+") if "@" in p]
    return ", ".join(dict.fromkeys(parts)) if parts else deck


def meta_table(entries: pd.DataFrame, min_share: float) -> pd.DataFrame:
    """Popularité de chaque deck (part des joueurs au deck connu) ; les decks rares deviennent « Autres »."""
    known = entries.dropna(subset=["deck"])
    if "deck_name_fr" not in known:
        known = known.assign(deck_name_fr=known["deck_name"])
    counts = known.groupby("deck").agg(name=("deck_name", "first"), name_fr=("deck_name_fr", "first"), players=("player", "size"))
    counts["share"] = counts["players"] / counts["players"].sum()
    threshold = max(min_share, MIN_PLAYERS / max(len(known), 1))
    main = counts[counts["share"] >= threshold].sort_values("players", ascending=False)
    for col in ("name", "name_fr"):  # deux versions d'un même deck : on précise les extensions
        twins = main[col].duplicated(keep=False)
        main.loc[twins, col] = [f"{n} ({versions(d)})" for n, d in zip(main.loc[twins, col], main.index[twins])]
    rest = counts[counts["share"] < threshold]
    if len(rest):
        others = pd.DataFrame(
            {"name": [OTHERS_NAME], "name_fr": [OTHERS_NAME], "players": [rest["players"].sum()], "share": [rest["share"].sum()]},
            index=pd.Index([OTHERS], name="deck"),
        )
        main = pd.concat([main, others])
    return main


def games_long(data: Data, legal: pd.Series, decks: pd.Index) -> pd.DataFrame:
    """Chaque match vu des deux côtés : deck, liste jouable ou non, adversaire, manches gagnées, manches.

    Avec le détail des manches, chaque manche compte ; sinon le match vaut une manche (un nul : une
    demi-victoire de chaque côté). Les miroirs et les decks inconnus sont écartés.
    """
    m = data.matches
    total = (m["games1"] + m["games2"]).to_numpy()
    result = m["result"].to_numpy()
    wins1 = np.where(total > 0, m["games1"], np.select([result == 1, result == 0], [1.0, 0.5], 0.0)).astype(float)
    n = np.where(total > 0, total, 1).astype(float)

    info = data.entries.assign(legal=legal.to_numpy()).set_index(["tournament_id", "player"])[["deck", "legal"]]
    info["deck"] = info["deck"].where(info["deck"].isin(decks) | info["deck"].isna(), OTHERS)

    def side(col: str) -> pd.DataFrame:
        keys = pd.MultiIndex.from_arrays([m["tournament_id"], m[col]])
        return info.reindex(keys).reset_index(drop=True)

    s1, s2 = side("player1"), side("player2")
    long = pd.concat(
        [
            pd.DataFrame({"deck": s1["deck"], "legal": s1["legal"], "opp": s2["deck"], "wins": wins1, "n": n}),
            pd.DataFrame({"deck": s2["deck"], "legal": s2["legal"], "opp": s1["deck"], "wins": n - wins1, "n": n}),
        ],
        ignore_index=True,
    )
    long = long.dropna(subset=["deck", "opp"])
    long = long[long["deck"] != long["opp"]]
    long["legal"] = long["legal"].fillna(False).astype(bool)
    return long


def pair_matrices(long: pd.DataFrame, decks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """W[i, j] = manches gagnées par i contre j ; N[i, j] = manches jouées entre i et j."""
    index = {d: i for i, d in enumerate(decks)}
    agg = long.groupby(["deck", "opp"])[["wins", "n"]].sum()
    W = np.zeros((len(decks), len(decks)))
    N = np.zeros_like(W)
    for (d, o), row in agg.iterrows():
        W[index[d], index[o]] = row["wins"]
        N[index[d], index[o]] = row["n"]
    return W, N


# --------------------------------------------------------------------------- modèles


BT_PRIOR_GAMES = 2.0


def bradley_terry(W: np.ndarray, N: np.ndarray, prior_games: float = BT_PRIOR_GAMES, iters: int = 2000, tol: float = 1e-10) -> np.ndarray:
    """Forces π (algorithme MM de Hunter, 2004), avec ``prior_games`` manches fictives à 50 % contre
    une référence de force 1 : la solution existe même pour un deck sans victoire."""
    wins = W.sum(axis=1) + prior_games / 2
    pi = np.ones(len(W))
    for _ in range(iters):
        denom = (N / (pi[:, None] + pi[None, :])).sum(axis=1) + prior_games / (pi + 1)
        new = wins / denom
        if np.max(np.abs(new - pi)) < tol:
            return new
        pi = new
    return pi


def bt_probabilities(pi: np.ndarray) -> np.ndarray:
    return pi[:, None] / (pi[:, None] + pi[None, :])


_lgamma = np.vectorize(math.lgamma, otypes=[float])


def fit_kappa(W: np.ndarray, N: np.ndarray, p0: np.ndarray, grid: np.ndarray | None = None) -> float:
    """κ qui maximise la vraisemblance bêta-binomiale des duels observés (Bayes empirique).

    Grand κ : les duels collent à Bradley-Terry (peu d'effet « pierre-feuille-ciseaux ») ;
    petit κ : les matchups s'écartent nettement de la force globale.
    """
    grid = np.logspace(0, 4, 161) if grid is None else grid
    iu = np.triu_indices(len(W), 1)
    w, n, p = W[iu], N[iu], p0[iu]
    keep = n > 0
    w, n, p = w[keep], n[keep], p[keep]
    if not len(n):
        return float(grid[-1])
    scores = []
    for k in grid:
        a, b = k * p, k * (1 - p)
        scores.append(np.sum(_lgamma(k) - _lgamma(a) - _lgamma(b) + _lgamma(w + a) + _lgamma(n - w + b) - _lgamma(n + k)))
    return float(grid[int(np.argmax(scores))])


@dataclass
class Model:
    meta: pd.DataFrame  # index deck : name, name_fr, players, share
    decks: list[str]
    W: np.ndarray
    N: np.ndarray
    pi: np.ndarray
    p0: np.ndarray
    kappa: float

    @property
    def posterior_mean(self) -> np.ndarray:
        p = (self.W + self.kappa * self.p0) / (self.N + self.kappa)
        np.fill_diagonal(p, 0.5)
        return p


def strength_se(model: Model) -> np.ndarray:
    """Écart-type approché de log π de chaque deck : inverse de la racine de l'information de Fisher
    (manches jouées pondérées par p(1 − p), plus les manches fictives de l'a priori)."""
    p = bt_probabilities(model.pi)
    info = (model.N * p * (1 - p)).sum(axis=1)
    p_ref = model.pi / (model.pi + 1)
    info = info + BT_PRIOR_GAMES * p_ref * (1 - p_ref)
    return 1 / np.sqrt(info)


def fit(data: Data, min_share: float = 0.005) -> tuple[Model, pd.DataFrame]:
    """Ajuste le modèle sur toutes les listes ; renvoie aussi la table des manches."""
    meta = meta_table(data.entries, min_share)
    decks = list(meta.index)
    long = games_long(data, legal_entries(data, None), meta.index)
    W, N = pair_matrices(long, decks)
    pi = bradley_terry(W, N)
    p0 = bt_probabilities(pi)
    return Model(meta, decks, W, N, pi, p0, fit_kappa(W, N, p0)), long


# --------------------------------------------------------------------------- classement


@dataclass
class Matchups:
    """Pour chaque deck candidat (lignes) contre chaque deck de la méta (colonnes)."""

    mean: pd.DataFrame  # probabilité de gagner une manche, a posteriori
    low: pd.DataFrame  # borne basse de l'intervalle à 90 %
    high: pd.DataFrame  # borne haute
    games: pd.DataFrame  # manches observées avec des listes jouables


def evaluate_matchups(
    model: Model,
    data: Data,
    legal: pd.Series,
    min_lists: int = 10,
    draws: int = 4000,
    seed: int = 0,
) -> tuple[pd.DataFrame, Matchups | None]:
    """Taux de victoire attendu de chaque deck jouable, avec intervalle, P(meilleur) et matchups.

    Pour un deck candidat i et un adversaire j, seules ses listes jouables comptent :
    a = victoires_jouables + κ·q, b = défaites_jouables + κ·(1 − q), où q est l'estimation tirée des
    autres listes de l'archétype, elle-même lissée vers Bradley-Terry. Sans filtre, il n'y a pas
    d'« autres listes » et l'on retrouve exactement le modèle complet.
    """
    decks, index, k = model.decks, {d: i for i, d in enumerate(model.decks)}, model.kappa
    long = games_long(data, legal, model.meta.index)
    entries = data.entries.assign(legal=legal.to_numpy())
    lists = entries[entries["legal"]].groupby("deck").size()
    candidates = [d for d in decks if d != OTHERS and lists.get(d, 0) >= min_lists]
    if not candidates:
        return pd.DataFrame(), None

    K, C = len(decks), len(candidates)
    rows_idx = np.array([index[d] for d in candidates])
    WL, NL, WC, NC = (np.zeros((C, K)) for _ in range(4))  # manches gagnées/jouées : listes jouables, autres listes
    for c, d in enumerate(candidates):
        rows = long[long["deck"] == d]
        for flag, (w_out, n_out) in ((True, (WL, NL)), (False, (WC, NC))):
            g = rows[rows["legal"] == flag].groupby("opp")[["wins", "n"]].sum()
            w_out[c], n_out[c] = g["wins"].reindex(decks, fill_value=0).to_numpy(), g["n"].reindex(decks, fill_value=0).to_numpy()

    def beta_params(p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Paramètres a, b des matchups des candidats pour une matrice p0 donnée (…, C, K)."""
        q = (WC + k * p0) / (NC + k)
        a, b = WL + k * q, NL - WL + k * (1 - q)
        a[..., np.arange(C), rows_idx] = b[..., np.arange(C), rows_idx] = 1e6  # miroir : 50 %
        return a, b

    A, B = beta_params(model.p0[rows_idx])
    share = model.meta["share"].to_numpy()
    mean_p = A / (A + B)
    expected = mean_p @ share

    # Monte-Carlo : forces de Bradley-Terry (loi normale approchée de log π), puis matchups (lois bêta)
    # et méta (loi de Dirichlet). Sans la première étape, un grand κ figerait les matchups sur une
    # prédiction de Bradley-Terry supposée exacte : intervalles bien trop étroits sur un petit format.
    rng = np.random.default_rng(seed)
    se = strength_se(model)
    e_draws = np.empty((draws, C))
    p_chunks = []
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        theta = np.log(model.pi)[None, :] + rng.standard_normal((size, K)) * se[None, :]
        p0_draws = 1 / (1 + np.exp(theta[:, None, :] - theta[:, rows_idx, None]))  # (tirages, C, K)
        a, b = beta_params(p0_draws)
        p = rng.beta(a, b)
        s = rng.dirichlet(model.meta["players"].to_numpy() + 0.5, size=size)
        e_draws[start: start + size] = np.einsum("sck,sk->sc", p, s)
        p_chunks.append(p.astype(np.float32))
    p_all = np.concatenate(p_chunks)
    p_low, p_high = np.quantile(p_all, [0.05, 0.95], axis=0)
    del p_all, p_chunks
    best = np.bincount(e_draws.argmax(axis=1), minlength=C) / draws
    G = NL

    n_games = np.array([long[(long["deck"] == d) & long["legal"]]["n"].sum() for d in candidates])
    major = share >= 0.02
    worst = []
    for c, d in enumerate(candidates):
        order = [j for j in np.argsort(mean_p[c]) if major[j] and decks[j] not in (d, OTHERS)][:3]
        worst.append(", ".join(f"{model.meta['name_fr'].iloc[j]} {mean_p[c, j]:.0%}" for j in order))

    out = pd.DataFrame(
        {
            "deck": candidates,
            "name": model.meta.loc[candidates, "name"].to_numpy(),
            "name_fr": model.meta.loc[candidates, "name_fr"].to_numpy(),
            "share": model.meta.loc[candidates, "share"].to_numpy(),
            "lists": [int(lists[d]) for d in candidates],
            "games": n_games,
            "win_rate": expected,
            "low": np.quantile(e_draws, 0.05, axis=0),
            "high": np.quantile(e_draws, 0.95, axis=0),
            "p_best": best,
            "worst": worst,
        }
    )
    frame = lambda x: pd.DataFrame(x, index=pd.Index(candidates, name="deck"), columns=pd.Index(decks, name="opp"))
    matchups = Matchups(frame(mean_p), frame(p_low), frame(p_high), frame(G))
    return out.sort_values("win_rate", ascending=False, ignore_index=True), matchups


def evaluate(model: Model, data: Data, legal: pd.Series, min_lists: int = 10, draws: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Classement seul (voir ``evaluate_matchups``)."""
    return evaluate_matchups(model, data, legal, min_lists, draws, seed)[0]


def matchup_table(analysis: "Analysis", deck: str, min_share: float = 0.0) -> pd.DataFrame:
    """Matchups d'un deck candidat contre chaque deck de la méta, du pire au meilleur."""
    m, meta = analysis.matchups, analysis.model.meta
    if m is None or deck not in m.mean.index:
        return pd.DataFrame()
    rows = pd.DataFrame(
        {
            "opp": m.mean.columns,
            "name": meta.loc[m.mean.columns, "name"].to_numpy(),
            "name_fr": meta.loc[m.mean.columns, "name_fr"].to_numpy(),
            "share": meta.loc[m.mean.columns, "share"].to_numpy(),
            "win": m.mean.loc[deck].to_numpy(),
            "low": m.low.loc[deck].to_numpy(),
            "high": m.high.loc[deck].to_numpy(),
            "games": m.games.loc[deck].to_numpy(),
        }
    )
    rows = rows[(rows["opp"] != deck) & (rows["share"] >= min_share)]
    return rows.sort_values("win", ignore_index=True)


def print_code(set_code: str, number) -> str:
    """Code d'une impression à la manière de Limitless : « C1-060 », « P-A-005 »."""
    short = set_code.replace("PROMO-", "P-") if isinstance(set_code, str) else set_code
    return f"{short}-{int(number):03d}"


def deck_list(data: Data, legal: pd.Series, deck: str, allowed=None) -> tuple[pd.DataFrame, int, int]:
    """Liste la plus jouée (à l'identique) parmi les listes jouables d'un deck.

    Renvoie (cartes, nombre de joueurs avec cette liste exacte, nombre de listes jouables). Pour
    chaque carte, la version proposée vient d'une extension permise (à défaut, la promo d'une carte
    permise en plus), puis la moins rare, puis la plus ancienne ; à date égale, une extension normale
    passe avant une promo (souvent une récompense d'événement passé).
    """
    pool = as_pool(allowed)
    entries = data.entries[legal.to_numpy() & (data.entries["deck"] == deck).to_numpy()]
    keys = entries[["tournament_id", "player"]]
    dc = data.deck_cards.merge(keys, on=["tournament_id", "player"])
    # 1 version normale + 1 version alternative d'une même carte = 2 exemplaires de cette carte
    dc = dc.groupby(["tournament_id", "player", "card_id"], as_index=False)["count"].sum().sort_values("card_id")
    dc["pair"] = list(zip(dc["card_id"], dc["count"]))
    signatures = dc.groupby(["tournament_id", "player"])["pair"].agg(tuple)
    if signatures.empty:
        return pd.DataFrame(), 0, 0
    (signature, count), = Counter(signatures).most_common(1)

    prints = data.prints
    rank = {r: i for i, r in enumerate(RARITY_ORDER)}
    release = dict(zip(data.sets["set"], data.sets["release_date"]))
    prints = prints.assign(
        _outside=pool.print_rank(prints) if pool is not None else 0,
        _promo=prints["set"].str.startswith("PROMO"),
        _rank=prints["rarity"].map(rank).fillna(99),
        _release=prints["set"].map(release),
    )
    suggested = prints.sort_values(["_outside", "_rank", "_release", "_promo", "number"]).drop_duplicates("card_id").set_index("card_id")

    cards = data.cards.set_index("card_id")
    has_fr = "name_fr" in cards
    ok = available_cards(data.cards, pool)
    rows = [
        {
            "count": n,
            "name": cards.loc[cid, "name"],
            "name_fr": cards.loc[cid, "name_fr"] if has_fr and isinstance(cards.loc[cid, "name_fr"], str) else cards.loc[cid, "name"],
            "category": cards.loc[cid, "category"],
            "print": print_code(suggested.loc[cid, "set"], suggested.loc[cid, "number"]) if cid in suggested.index else "",
            "available": bool(ok.get(cid, False)),
            "card_id": cid,
        }
        for cid, n in signature
    ]
    table = pd.DataFrame(rows)
    table["_pokemon"] = table["category"] != "pokemon"
    table = table.sort_values(["_pokemon", "count", "name"], ascending=[True, False, True]).drop(columns="_pokemon")
    return table.reset_index(drop=True), count, len(signatures)


# --------------------------------------------------------------------------- validation


def holdout_scores(data: Data, split: str, min_share: float = 0.005, bootstrap: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Compare les prédictions de matchups sur des tournois postérieurs à ``split``.

    Modèles appris avant ``split`` : pile ou face (50 %), taux brut observé, Bradley-Terry seul, et
    Bradley-Terry + lissage bayésien. Scores par manche : perte logarithmique et score de Brier
    (plus bas = meilleur). ``attrs["gaps"]`` : écart de perte logarithmique par manche du modèle retenu
    avec chaque autre modèle, et son intervalle à 95 % par bootstrap sur les tournois de test.
    """
    t = data.tournaments
    train_ids = set(t.loc[t["date"] < split, "tournament_id"])
    train = replace(
        data,
        entries=data.entries[data.entries["tournament_id"].isin(train_ids)],
        matches=data.matches[data.matches["tournament_id"].isin(train_ids)],
    )
    test = replace(
        data,
        entries=data.entries[~data.entries["tournament_id"].isin(train_ids)],
        matches=data.matches[~data.matches["tournament_id"].isin(train_ids)],
    )
    model, _ = fit(train, min_share)
    long = games_long(test, legal_entries(test, None), model.meta.index)
    Wt, Nt = pair_matrices(long, model.decks)
    raw = np.where(model.N > 0, model.W / np.maximum(model.N, 1), 0.5)
    candidates = {
        "Pile ou face (50 %)": np.full_like(raw, 0.5),
        "Taux brut observé": raw,
        "Bradley-Terry seul": model.p0,
        "Bradley-Terry + lissage bayésien": model.posterior_mean,
    }
    iu = np.triu_indices(len(Wt), 1)
    w, n = Wt[iu], Nt[iu]
    rows = []
    for name, p in candidates.items():
        q = np.clip(p[iu], 0.01, 0.99)
        logloss = -(w * np.log(q) + (n - w) * np.log(1 - q)).sum() / n.sum()
        brier = (w * (1 - q) ** 2 + (n - w) * q**2).sum() / n.sum()
        rows.append({"modèle": name, "log_loss": logloss, "brier": brier})
    out = pd.DataFrame(rows)

    # Écart de perte logarithmique du modèle retenu avec chaque autre modèle, et son intervalle à 95 %
    # par bootstrap sur les tournois de test (les manches d'un même tournoi ne sont pas indépendantes)
    index = {d: i for i, d in enumerate(model.decks)}
    i, j = long["deck"].map(index).to_numpy(), long["opp"].map(index).to_numpy()
    # games_long empile les deux côtés de chaque match : son index renvoie au match d'origine
    tid = np.concatenate([test.matches["tournament_id"].to_numpy()] * 2)[long.index.to_numpy()]
    half = i < j  # chaque match une seule fois
    per_t = pd.DataFrame({"t": tid[half], "n": long["n"].to_numpy()[half]})
    for name, p in candidates.items():
        q = np.clip(p[i[half], j[half]], 0.01, 0.99)
        wins = long["wins"].to_numpy()[half]
        per_t[name] = -(wins * np.log(q) + (per_t["n"] - wins) * np.log(1 - q))
    sums = per_t.groupby("t").sum()
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(sums), size=(bootstrap, len(sums)))
    best = "Bradley-Terry + lissage bayésien"
    gaps = {}
    for name in candidates:
        if name == best:
            continue
        diff = (sums[best] - sums[name]).to_numpy()
        values = diff[boot].sum(axis=1) / sums["n"].to_numpy()[boot].sum(axis=1)
        gaps[name] = (diff.sum() / sums["n"].sum(), *np.quantile(values, [0.025, 0.975]))
    out.attrs.update(kappa=model.kappa, test_games=float(n.sum()), decks=len(model.decks), test_tournaments=len(sums), gaps=gaps)
    return out


# --------------------------------------------------------------------------- tout-en-un


@dataclass
class Analysis:
    data: Data
    model: Model
    legal: pd.Series
    allowed: Pool | None
    ranking: pd.DataFrame
    matchups: Matchups | None = None
    format_id: str = STANDARD
    since: str | None = None


def analyze(
    data: Data,
    sets: set[str] | None = None,
    event_promos: bool = False,
    since: str | None = None,
    max_missing: int = 0,
    min_share: float = 0.005,
    min_lists: int = 10,
    draws: int = 4000,
    seed: int = 0,
    format_id: str = STANDARD,
    pool: Pool | None = None,
) -> Analysis:
    """Classement des decks d'un format (par défaut le standard, pour le classé).

    La réserve de son deck est ``pool`` si elle est donnée ; sinon, en standard, les extensions
    ``sets`` (None : toutes), les dresseurs de boutique, et les promos d'événement si ``event_promos`` ;
    dans un autre format, la réserve du format. ``max_missing`` : exemplaires hors réserve tolérés dans
    une liste (à trouver ou remplacer).
    """
    since = since or default_since(data, format_id)
    window = select(data, since, format_id=format_id)
    model, _ = fit(window, min_share)
    if pool is None:
        pool = user_pool(data, sets, event_promos) if format_id == STANDARD else rules_pool(data, format_id)
    legal = legal_entries(window, pool, max_missing)
    ranking, matchups = evaluate_matchups(model, window, legal, min_lists=min_lists, draws=draws, seed=seed)
    return Analysis(window, model, legal, pool, ranking, matchups, format_id, since)
