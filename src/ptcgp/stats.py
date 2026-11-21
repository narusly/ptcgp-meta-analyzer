"""Statistiques de méta pour le classé : popularité, matchups lissés et taux de victoire attendu.

Méthode (détaillée dans le README) :

1. L'unité d'observation est la manche, car le classé se joue en une manche. Un nul vaut une
   demi-victoire. Les decks trop rares (moins de ``min_share`` des joueurs) sont regroupés en
   « Autres decks ».
2. Force globale de chaque deck : modèle de Bradley-Terry, P(A bat B) = πA / (πA + πB).
3. Matchups : le résultat observé de A contre B est lissé vers la prédiction de Bradley-Terry, avec
   un poids κ (en manches fictives) estimé par Bayes empirique sur la vraisemblance bêta-binomiale.
   A posteriori, P(A bat B) suit une loi Bêta(victoires + κ·p0, défaites + κ·(1 − p0)).
4. Taux de victoire attendu en classé : moyenne des matchups pondérée par la popularité des decks
   adverses (miroir compté à 50 %).
5. Incertitude : tirages de Monte-Carlo (lois bêta des matchups, loi de Dirichlet de la méta),
   d'où un intervalle à 90 % et la probabilité d'être le meilleur deck.

Le filtre par extension ne s'applique qu'à son propre deck : seules les listes jouables avec les
extensions choisies comptent pour le deck candidat, et ses adversaires gardent la vraie méta. Les
autres listes du même archétype servent alors d'a priori (voir ``evaluate``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .archetypes import add_decks

OTHERS = "autres"
OTHERS_NAME = "Autres decks"
MAIN_SET = re.compile(r"^[A-Z]\d+[a-z]?$")
RARITY_ORDER = ["C", "U", "R", "RR", "AR", "SR", "SAR", "IM", "S", "SSR", "UR", "AN"]


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


def load(processed: Path) -> Data:
    cards = pd.read_csv(processed / "cards.csv")
    entries = pd.read_csv(processed / "entries.csv.gz")
    return Data(
        sets=pd.read_csv(processed / "sets.csv"),
        prints=pd.read_csv(processed / "prints.csv"),
        cards=cards,
        tournaments=pd.read_csv(processed / "tournaments.csv"),
        entries=add_decks(entries, cards.loc[cards["category"] == "pokemon", "name"]),
        deck_cards=pd.read_csv(processed / "deck_cards.csv.gz"),
        matches=pd.read_csv(processed / "matches.csv.gz"),
    )


def latest_expansion(sets: pd.DataFrame) -> pd.Series:
    """Dernière extension principale sortie (les promos et packs spéciaux sont ignorés)."""
    main = sets[sets["set"].str.match(MAIN_SET)]
    return main.sort_values("release_date").iloc[-1]


def select(data: Data, since: str | None = None, min_records_ok: float = 0.95) -> Data:
    """Fenêtre de temps (par défaut : depuis la dernière extension) et tournois au résultat fiable.

    ``min_records_ok`` écarte les tournois dont les bilans recalculés ne collent pas, comme les
    tournois par équipes que l'API classe en format standard.
    """
    since = since or latest_expansion(data.sets)["release_date"]
    t = data.tournaments
    kept = t[(t["date"] >= since) & (t["records_ok"] >= min_records_ok)]
    ids = set(kept["tournament_id"])
    return replace(
        data,
        tournaments=kept,
        entries=data.entries[data.entries["tournament_id"].isin(ids)],
        deck_cards=data.deck_cards[data.deck_cards["tournament_id"].isin(ids)],
        matches=data.matches[data.matches["tournament_id"].isin(ids)],
    )


def promo_sets(sets: pd.DataFrame) -> set[str]:
    return {s for s in sets["set"] if s.startswith("PROMO")}


def available_cards(cards: pd.DataFrame, allowed: set[str] | None) -> pd.Series:
    """True pour les cartes dont au moins une impression vient d'une extension autorisée."""
    if allowed is None:
        return pd.Series(True, index=cards["card_id"])
    return cards.set_index("card_id")["sets"].str.split("|").map(lambda s: bool(set(s) & allowed))


def missing_copies(data: Data, allowed: set[str] | None) -> pd.Series:
    """Nombre d'exemplaires de la decklist introuvables dans les extensions ``allowed`` (0 si tout est jouable)."""
    ok = available_cards(data.cards, allowed)
    dc = data.deck_cards
    missing = dc["count"].where(~dc["card_id"].map(ok).fillna(False).astype(bool), 0)
    per_entry = missing.groupby([dc["tournament_id"], dc["player"]]).sum()
    keys = pd.MultiIndex.from_frame(data.entries[["tournament_id", "player"]])
    return pd.Series(per_entry.reindex(keys).fillna(0).to_numpy(), index=data.entries.index)


def legal_entries(data: Data, allowed: set[str] | None, max_missing: int = 0) -> pd.Series:
    """True pour les joueurs dont la decklist est jouable avec les extensions ``allowed``.

    ``max_missing`` tolère quelques exemplaires venant d'autres extensions (à trouver ou à remplacer) :
    les decks de méta mélangent beaucoup d'extensions, et un filtre strict sur une ou deux
    extensions ne laisse souvent aucun deck. ``allowed=None`` : toutes les extensions.
    """
    has_list = data.entries["has_decklist"].astype(bool)
    if allowed is None:
        return has_list
    return has_list & (missing_copies(data, allowed) <= max_missing)


# --------------------------------------------------------------------------- méta et manches


def meta_table(entries: pd.DataFrame, min_share: float) -> pd.DataFrame:
    """Popularité de chaque deck (part des joueurs au deck connu) ; les decks rares deviennent « Autres »."""
    known = entries.dropna(subset=["deck"])
    counts = known.groupby("deck").agg(name=("deck_name", "first"), players=("player", "size"))
    counts["share"] = counts["players"] / counts["players"].sum()
    main = counts[counts["share"] >= min_share].sort_values("players", ascending=False)
    rest = counts[counts["share"] < min_share]
    if len(rest):
        others = pd.DataFrame(
            {"name": [OTHERS_NAME], "players": [rest["players"].sum()], "share": [rest["share"].sum()]},
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

    info = data.entries.assign(legal=legal).set_index(["tournament_id", "player"])[["deck", "legal"]]
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


def bradley_terry(W: np.ndarray, N: np.ndarray, prior_games: float = 2.0, iters: int = 2000, tol: float = 1e-10) -> np.ndarray:
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
    meta: pd.DataFrame  # index deck : name, players, share
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


def evaluate(
    model: Model,
    data: Data,
    legal: pd.Series,
    min_lists: int = 10,
    draws: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """Taux de victoire attendu en classé de chaque deck jouable, avec intervalle et P(meilleur).

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
        return pd.DataFrame()

    K = len(decks)
    A = np.zeros((len(candidates), K))
    B = np.zeros_like(A)
    for c, d in enumerate(candidates):
        i = index[d]
        rows = long[long["deck"] == d]
        stats = {}
        for flag in (True, False):
            g = rows[rows["legal"] == flag].groupby("opp")[["wins", "n"]].sum()
            stats[flag] = (g["wins"].reindex(decks, fill_value=0).to_numpy(), g["n"].reindex(decks, fill_value=0).to_numpy())
        (wl, nl), (wc, nc) = stats[True], stats[False]
        q = (wc + k * model.p0[i]) / (nc + k)
        A[c], B[c] = wl + k * q, nl - wl + k * (1 - q)
        A[c, i] = B[c, i] = 1e6  # miroir : 50 %

    share = model.meta["share"].to_numpy()
    mean_p = A / (A + B)
    expected = mean_p @ share

    rng = np.random.default_rng(seed)
    p_draws = rng.beta(A[None], B[None], size=(draws, *A.shape))
    s_draws = rng.dirichlet(model.meta["players"].to_numpy() + 0.5, size=draws)
    e_draws = np.einsum("sck,sk->sc", p_draws, s_draws)
    best = np.bincount(e_draws.argmax(axis=1), minlength=len(candidates)) / draws

    n_games = np.array([long[(long["deck"] == d) & long["legal"]]["n"].sum() for d in candidates])
    major = share >= 0.02
    worst = []
    for c, d in enumerate(candidates):
        order = [j for j in np.argsort(mean_p[c]) if major[j] and decks[j] not in (d, OTHERS)][:3]
        worst.append(", ".join(f"{model.meta['name'].iloc[j]} {mean_p[c, j]:.0%}" for j in order))

    out = pd.DataFrame(
        {
            "deck": candidates,
            "name": model.meta.loc[candidates, "name"].to_numpy(),
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
    return out.sort_values("win_rate", ascending=False, ignore_index=True)


def deck_list(data: Data, legal: pd.Series, deck: str, allowed: set[str] | None = None) -> tuple[pd.DataFrame, int, int]:
    """Liste la plus jouée (à l'identique) parmi les listes jouables d'un deck.

    Renvoie (cartes, nombre de joueurs avec cette liste exacte, nombre de listes jouables). Pour
    chaque carte, la version proposée est la moins rare parmi les extensions autorisées, puis la plus
    ancienne ; à date égale, une extension normale passe avant une promo (souvent une récompense
    d'événement passé).
    """
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
    if allowed is not None:  # version autorisée si elle existe, sinon n'importe laquelle (carte manquante)
        prints = prints.assign(_outside=~prints["set"].isin(allowed)).sort_values("_outside", kind="stable")
    rank = {r: i for i, r in enumerate(RARITY_ORDER)}
    release = dict(zip(data.sets["set"], data.sets["release_date"]))
    prints = prints.assign(
        _promo=prints["set"].str.startswith("PROMO"),
        _rank=prints["rarity"].map(rank).fillna(99),
        _release=prints["set"].map(release),
    )
    order = (["_outside"] if "_outside" in prints else []) + ["_rank", "_release", "_promo", "number"]
    suggested = prints.sort_values(order).drop_duplicates("card_id").set_index("card_id")

    cards = data.cards.set_index("card_id")
    ok = available_cards(data.cards, allowed)
    rows = [
        {
            "count": n,
            "name": cards.loc[cid, "name"],
            "category": cards.loc[cid, "category"],
            "print": f"{suggested.loc[cid, 'set']}-{int(suggested.loc[cid, 'number']):03d}" if cid in suggested.index else "",
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


def holdout_scores(data: Data, split: str, min_share: float = 0.005) -> pd.DataFrame:
    """Compare les prédictions de matchups sur des tournois postérieurs à ``split``.

    Modèles appris avant ``split`` : pile ou face (50 %), taux brut observé, Bradley-Terry seul, et
    Bradley-Terry + lissage bayésien. Scores par manche : perte logarithmique et score de Brier
    (plus bas = meilleur).
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
    out.attrs.update(kappa=model.kappa, test_games=float(n.sum()), decks=len(model.decks))
    return out


# --------------------------------------------------------------------------- tout-en-un


@dataclass
class Analysis:
    data: Data
    model: Model
    legal: pd.Series
    allowed: set[str] | None
    ranking: pd.DataFrame


def analyze(
    data: Data,
    sets: set[str] | None = None,
    include_promos: bool = True,
    since: str | None = None,
    max_missing: int = 0,
    min_share: float = 0.005,
    min_lists: int = 10,
    draws: int = 4000,
    seed: int = 0,
) -> Analysis:
    """Classement des decks pour le classé, avec les extensions ``sets`` (None : toutes).

    ``max_missing`` : exemplaires hors de ces extensions tolérés dans une liste (à trouver ou remplacer).
    """
    window = select(data, since)
    model, _ = fit(window, min_share)
    allowed = None if sets is None else set(sets) | (promo_sets(data.sets) if include_promos else set())
    legal = legal_entries(window, allowed, max_missing)
    ranking = evaluate(model, window, legal, min_lists=min_lists, draws=draws, seed=seed)
    return Analysis(window, model, legal, allowed, ranking)
