"""Formats à règles spéciales : réserve de cartes de chaque tournoi, puis regroupement par format.

Limitless ne publie pas la réserve de cartes d'un tournoi CUSTOM (seulement d'éventuelles cartes
bannies) : on la déduit de ses decklists.

1. **Extensions utilisées.** Couverture gloutonne des cartes jouées : d'abord par les extensions
   normales, puis par les compilations (les Booster de Luxe ne contiennent que des réimpressions et
   « expliqueraient » sinon à eux seuls les cartes de dix extensions). Une extension n'est retenue que
   si elle apporte au moins ``min_cards`` cartes jouées. Les cartes qui restent sont les **cartes
   permises en plus** (souvent des promos d'événement). Les dresseurs de boutique sont toujours permis.
2. **Restriction réelle.** Une extension non retenue est **absente** si ses cartes (hors extensions
   retenues) sont nettement moins jouées que dans le standard de la même période : test binomial
   unilatéral (p < ``alpha``) et taux observé inférieur au tiers du taux standard. Une extension sortie
   depuis moins de ``grace_days`` jours n'est pas testée (les joueurs ne l'ont pas encore adoptée).
   Même test pour les cartes ex : un tournoi sans aucun ex suit un format « sans ex ».
3. **Formats.** Un tournoi à réserve restreinte (au moins une extension absente, ou aucun ex) rejoint
   le format compatible le plus proche, déjà fondé par un tournoi plus grand ; sinon il en fonde un.
   Les tournois CUSTOM à réserve complète sont écartés : leur règle (Dual Deck, mono-type, sans pile ou
   face…) ne se voit pas dans les decklists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

STANDARD = "standard"


# --------------------------------------------------------------------------- réserve de cartes


@dataclass(frozen=True)
class Pool:
    """Cartes jouables : extensions permises, cartes permises en plus, cartes retirées, ex ou non.

    ``sets=None`` : toutes les extensions. Les cartes de ``always`` (dresseurs de boutique) sont
    toujours permises, sauf si on les retire.
    """

    sets: frozenset[str] | None = None
    cards: frozenset[str] = frozenset()
    removed: frozenset[str] = frozenset()
    no_ex: bool = False
    always: frozenset[str] = frozenset()

    def available(self, cards: pd.DataFrame) -> pd.Series:
        """True pour chaque carte jouable (index : ``card_id``)."""
        c = cards.set_index("card_id")
        if self.sets is None:
            ok = pd.Series(True, index=c.index)
        else:
            ok = c["sets"].fillna("").str.split("|").map(lambda s: bool(set(s) & self.sets))
        ok = ok | c.index.isin(self.cards | self.always)
        if self.no_ex and "is_ex" in c:
            ok = ok & ~c["is_ex"].fillna(False).astype(bool)
        return ok & ~c.index.isin(self.removed)

    def print_rank(self, prints: pd.DataFrame) -> pd.Series:
        """Préférence pour la version à proposer : 0 = extension permise, ou version boutique d'un
        dresseur de boutique ; 1 = promo d'une carte permise en plus (c'est ainsi qu'on l'obtient) ;
        2 = autre version."""
        promo = prints["set"].str.startswith("PROMO")
        in_sets = prints["set"].isin(self.sets) if self.sets is not None else pd.Series(True, index=prints.index)
        shop = prints["card_id"].isin(self.always) & promo
        listed = prints["card_id"].isin(self.cards) & promo
        return pd.Series(np.select([shop | (in_sets & ~prints["card_id"].isin(self.always)), listed], [0, 1], 2), index=prints.index)


def as_pool(allowed) -> Pool | None:
    """Accepte une réserve, un ensemble de codes d'extension, ou None (toutes les cartes)."""
    if allowed is None or isinstance(allowed, Pool):
        return allowed
    return Pool(sets=frozenset(allowed))


# --------------------------------------------------------------------------- outils


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) pour X ~ Binomiale(n, p), calculée en logarithmes."""
    if k >= n:
        return 1.0
    p = min(max(p, 1e-12), 1 - 1e-12)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(x - top) for x in logs))


def main_sets(sets: pd.DataFrame) -> pd.DataFrame:
    """Extensions hors promos, dans l'ordre de sortie (les promos sont des cartes permises une à une)."""
    return sets[~sets["set"].str.startswith("PROMO")].sort_values(["release_date", "set"], ignore_index=True)


def compilations(cards: pd.DataFrame, sets: pd.DataFrame, threshold: float = 0.9) -> set[str]:
    """Extensions faites (presque) uniquement de réimpressions : les Booster de Luxe A4b et B4b."""
    first = cards.set_index("card_id")["first_set"]
    pairs = cards[["card_id", "sets"]].assign(set=cards["sets"].str.split("|")).explode("set")
    pairs["reprint"] = pairs["card_id"].map(first) != pairs["set"]
    share = pairs.groupby("set")["reprint"].mean()
    return {s for s in main_sets(sets)["set"] if share.get(s, 0) >= threshold}


# --------------------------------------------------------------------------- listes en matrice


@dataclass
class ListMatrix:
    """Une ligne par decklist, une colonne par carte jouée quelque part : True si la carte y figure."""

    M: np.ndarray
    tournament: np.ndarray  # tournoi de chaque ligne
    cards: pd.Index  # card_id de chaque colonne
    duplicates: np.ndarray  # la liste joue-t-elle une carte en au moins 2 exemplaires ?

    @classmethod
    def build(cls, deck_cards: pd.DataFrame) -> "ListMatrix":
        dc = deck_cards.dropna(subset=["card_id"])
        # 1 version normale + 1 version alternative d'une même carte = 2 exemplaires
        dc = dc.groupby(["tournament_id", "player", "card_id"], as_index=False, sort=False)["count"].sum()
        rows, lists = pd.factorize(pd.MultiIndex.from_frame(dc[["tournament_id", "player"]]))
        cols, cards = pd.factorize(dc["card_id"])
        M = np.zeros((len(lists), len(cards)), dtype=bool)
        M[rows, cols] = True
        dup = np.zeros(len(lists), dtype=bool)
        dup[rows[dc["count"].to_numpy() >= 2]] = True
        return cls(M, np.asarray(lists.get_level_values(0)), pd.Index(cards), dup)

    def mask(self, card_ids) -> np.ndarray:
        return self.cards.isin(list(card_ids))


# --------------------------------------------------------------------------- déduction par tournoi


@dataclass
class TournamentPool:
    tournament_id: str
    lists: int
    sets: tuple[str, ...]  # extensions de la réserve (retenues, sous-entendues ou entre deux retenues)
    used_sets: tuple[str, ...]  # extensions retenues par la couverture
    extras: dict[str, int]  # carte permise en plus -> nombre de listes qui la jouent
    absent: tuple[str, ...]  # extensions nettement absentes
    no_ex: bool
    singleton: bool  # un seul exemplaire par carte
    restricted: bool


@dataclass
class Inference:
    """Contexte commun : cartes par extension, standard de référence, dresseurs de boutique."""

    cards: pd.DataFrame
    sets: pd.DataFrame
    matrix: ListMatrix
    standard_lists: pd.DataFrame  # tournament_id, date des decklists standard de référence
    shop: set[str]
    min_cards: int = 4
    alpha: float = 1e-6  # environ 2 600 tests (tournois × extensions) : seuil strict
    grace_days: int = 7
    set_cards: dict[str, set[str]] = field(init=False)

    def __post_init__(self):
        exploded = self.cards[["card_id", "sets"]].assign(set=self.cards["sets"].str.split("|")).explode("set")
        self.set_cards = exploded.groupby("set")["card_id"].agg(set).to_dict()
        self.promo_cards = set(exploded.loc[exploded["set"].str.startswith("PROMO"), "card_id"])
        self.main = main_sets(self.sets)
        self.series = dict(zip(self.main["set"], self.main["series"]))
        self.compilations = compilations(self.cards, self.sets)
        self.ex_cards = set(self.cards.loc[self.cards["is_ex"].fillna(False).astype(bool), "card_id"])
        self.release = dict(zip(self.sets["set"], self.sets["release_date"]))

    # ----------------------------------------------------------------- standard de référence

    def reference_rows(self, date: str) -> np.ndarray:
        """Decklists standard de la même période (entre deux sorties d'extension) ; à défaut, ±30 jours."""
        releases = sorted(set(self.main["release_date"]))
        start = max([r for r in releases if r <= date[:10]], default=releases[0])
        later = [r for r in releases if r > start]
        end = later[0] if later else "9999"
        ref = self.standard_lists
        ids = set(ref.loc[(ref["date"].str[:10] >= start) & (ref["date"].str[:10] < end), "tournament_id"])
        if len(ref[ref["tournament_id"].isin(ids)]) < 200:
            day = pd.Timestamp(date[:10])
            lo, hi = str((day - timedelta(days=30)).date()), str((day + timedelta(days=30)).date())
            ids = set(ref.loc[(ref["date"].str[:10] >= lo) & (ref["date"].str[:10] <= hi), "tournament_id"])
        return np.isin(self.matrix.tournament, list(ids))

    # ----------------------------------------------------------------- un tournoi

    def infer(self, tournament_id: str, date: str) -> TournamentPool | None:
        mx = self.matrix
        rows = mx.tournament == tournament_id
        n = int(rows.sum())
        if n == 0:
            return None
        own = mx.M[rows]
        usage = pd.Series(own.sum(axis=0), index=mx.cards)
        used = set(usage[usage > 0].index)
        candidates = [s for s in self.main["set"] if self.release[s] <= date[:10]]
        ref_rows = self.reference_rows(date)
        reference = mx.M[ref_rows]

        # 1. Couverture gloutonne : extensions normales, puis compilations. Une carte qui existe aussi en
        #    promo ne suffit pas à retenir une extension : la promo peut être permise seule (une carte A2
        #    rééditée en PROMO-B dans un format série B).
        uncovered = used - self.shop
        chosen: list[str] = []
        for group in ([s for s in candidates if s not in self.compilations], [s for s in candidates if s in self.compilations]):
            while True:
                gains = {}
                for s in group:
                    if s not in chosen:
                        new = (uncovered & self.set_cards.get(s, set())) - self.promo_cards
                        gains[s] = (len(new), usage.reindex(list(new)).sum())
                if not gains:
                    break
                best = max(gains, key=gains.get)
                if gains[best][0] < self.min_cards:
                    break
                chosen.append(best)
                uncovered -= self.set_cards.get(best, set())

        # 2. Réserve et extensions absentes. Une extension retenue dont toutes les cartes jouées se
        #    trouvent déjà dans le reste de la réserve est retirée (réimpressions : une carte A2 rééditée
        #    en B4a ne fait pas d'un format série B un format A2 + B).
        pool, absent = self._pool(chosen, candidates, date, own, reference)
        pruned = True
        while pruned:
            pruned = False
            for s in reversed(chosen):
                # Une compilation (réimpressions) ne rend jamais redondante l'extension d'origine
                others = set().union(*(self.set_cards.get(x, set()) for x in pool if x != s and x not in self.compilations))
                if (used & self.set_cards.get(s, set())) - self.shop - self.promo_cards <= others:
                    chosen = [x for x in chosen if x != s]
                    pool, absent = self._pool(chosen, candidates, date, own, reference)
                    pruned = True
                    break

        # 3. Une vraie restriction par extension laisse, dans chaque série, une suite sans trou
        #    (A1 à A3b, C1 seule, série B). Un trou (B2a et B3a absentes entre B1 et B4a) trahit une
        #    autre règle, sur les Pokémon ou les types (« Unova Unleashed ») : on ne la retient pas.
        if absent and not self._contiguous(pool, absent):
            absent = []
        pool_sets = tuple(pool) if absent else tuple(candidates)
        covered = set().union(*(self.set_cards.get(s, set()) for s in pool_sets))
        uncovered = used - self.shop - covered

        # 4. Règles visibles dans les listes : aucun des ex de la réserve, un seul exemplaire par carte
        no_ex = self._rule_absent(own, reference, mx.mask(self.ex_cards & covered))
        dup, ref_dup = mx.duplicates[rows], mx.duplicates[ref_rows]
        singleton = bool(len(ref_dup)) and dup.sum() <= 0.02 * n and self._absent(int(dup.sum()), n, float(ref_dup.mean()))
        extras = {c: int(usage[c]) for c in sorted(uncovered, key=lambda c: -usage[c])}
        restricted = bool(absent) or no_ex or singleton
        return TournamentPool(tournament_id, n, pool_sets, tuple(chosen), extras, tuple(absent), no_ex, singleton, restricted)

    def _pool(self, chosen: list[str], candidates: list[str], date: str, own: np.ndarray, reference: np.ndarray) -> tuple[list[str], list[str]]:
        """Réserve et extensions absentes, en deux passages : les extensions situées entre deux
        extensions retenues et non absentes entrent dans la réserve, puis on reteste les autres.

        Les ex sont ignorés ici, pour ne pas confondre « peu d'ex » (1 Ex Challenge) et « extension
        absente ». Les extensions d'une même série sont aussi testées ensemble (plus puissant sur un
        petit tournoi).
        """
        order = list(self.main["set"])
        positions = [order.index(s) for s in chosen]
        pool, absent = list(chosen), []
        for _ in range(2):
            covered = set().union(*(self.set_cards.get(s, set()) for s in pool))
            implied = [s for s in candidates if s not in chosen and self.set_cards.get(s) and self.set_cards[s] <= covered | self.shop]
            rest = [s for s in candidates if s not in pool and s not in implied and self._testable(s, date)]
            absent = [s for s in rest if self._cards_absent(self.set_cards.get(s, set()), own, reference, covered)]
            for series in dict.fromkeys(self.series[s] for s in rest):
                group = [s for s in rest if self.series[s] == series and s not in absent]
                union = set().union(*(self.set_cards.get(s, set()) for s in group)) if group else set()
                if len(group) >= 2 and self._cards_absent(union, own, reference, covered):
                    absent += group
            between = [s for s in order[min(positions): max(positions) + 1] if s in candidates and s not in absent] if positions else []
            pool = [s for s in order if s in set(chosen) | set(between) | set(implied)]
        return pool, [s for s in order if s in absent]

    def _contiguous(self, pool: list[str], absent: list[str]) -> bool:
        """Aucune extension absente entre deux extensions de la réserve d'une même série."""
        for series in set(self.series[s] for s in pool):
            order = [s for s in self.main["set"] if self.series[s] == series]
            idx = [order.index(s) for s in pool if self.series[s] == series]
            if any(min(idx) < order.index(s) < max(idx) for s in absent if self.series[s] == series):
                return False
        return True

    def _testable(self, s: str, date: str) -> bool:
        """Une extension trop récente n'est pas testée : les joueurs ne l'ont pas encore adoptée."""
        return pd.Timestamp(date[:10]) - pd.Timestamp(self.release[s]) >= timedelta(days=self.grace_days)

    def _cards_absent(self, ids: set[str], own: np.ndarray, reference: np.ndarray, covered: set[str]) -> bool:
        """Les cartes ``ids`` (hors réserve, boutique et ex) sont-elles nettement moins jouées qu'en standard ?"""
        if not len(reference):
            return False
        cols = self.matrix.mask(ids - covered - self.shop - self.ex_cards)
        if not cols.any():
            return False
        return self._absent(int(own[:, cols].any(axis=1).sum()), len(own), float(reference[:, cols].any(axis=1).mean()))

    def _rule_absent(self, own: np.ndarray, reference: np.ndarray, cols: np.ndarray) -> bool:
        """Cartes (les ex…) quasi absentes des listes alors que le standard en joue beaucoup."""
        if not len(reference) or not cols.any():
            return False
        k = int(own[:, cols].any(axis=1).sum())
        return k <= 0.02 * len(own) and self._absent(k, len(own), float(reference[:, cols].any(axis=1).mean()))

    def _absent(self, k: int, n: int, rate: float) -> bool:
        return rate * n >= 5 and k / n < rate / 3 and binom_cdf(int(k), n, float(rate)) < self.alpha


# --------------------------------------------------------------------------- regroupement


@dataclass
class Format:
    format_id: str
    sets: frozenset[str] | None
    no_ex: bool
    singleton: bool
    members: list[TournamentPool]


def group(pools: list[TournamentPool]) -> list[Format]:
    """Chaque tournoi restreint rejoint le format compatible le plus proche (fondé par un plus grand)."""
    formats: list[Format] = []
    for p in sorted(pools, key=lambda p: (-p.lists, p.tournament_id)):
        sets = frozenset(p.sets) if p.absent else None
        best, best_gap = None, None
        for f in formats:
            if f.no_ex != p.no_ex or f.singleton != p.singleton or (f.sets is None) != (sets is None):
                continue
            if f.sets is not None and (not set(p.used_sets) <= f.sets or set(p.absent) & f.sets):
                continue
            gap = len(f.sets - sets) + len(sets - f.sets) if f.sets is not None else 0
            if best is None or gap < best_gap:
                best, best_gap = f, gap
        if best is not None:
            best.members.append(p)
        else:
            formats.append(Format(format_id(sets, p.no_ex, p.singleton), sets, p.no_ex, p.singleton, [p]))
    return formats


def format_id(sets: frozenset[str] | None, no_ex: bool, singleton: bool = False) -> str:
    base = "toutes" if sets is None else "_".join(sorted(sets))
    return base + ("-sans-ex" if no_ex else "") + ("-1-exemplaire" if singleton else "")


def label(fmt_sets: frozenset[str] | None, no_ex: bool, sets: pd.DataFrame, singleton: bool = False) -> str:
    """Nom lisible : « Fusion Double (C1) seule », « De A1 à A3b », « Série B (B1 à B4b) »…"""
    if fmt_sets is None:
        text = "Toutes les extensions"
    else:
        main = main_sets(sets)
        order = list(main["set"])
        chosen = [s for s in order if s in fmt_sets]
        names = dict(zip(main["set"], main["name_fr"].where(main["name_fr"].fillna("") != "", main["name"])))
        positions = [order.index(s) for s in chosen]
        contiguous = positions == list(range(positions[0], positions[-1] + 1))
        series = main.groupby("series")["set"].agg(list).to_dict()
        whole = [k for k, v in series.items() if [s for s in v if s in fmt_sets] == v and len(v) == len(chosen)]
        if len(chosen) == 1:
            text = f"{names[chosen[0]]} ({chosen[0]}) seule"
        elif whole:
            text = f"Série {whole[0]} ({chosen[0]} à {chosen[-1]})"
        elif contiguous:
            text = f"De {chosen[0]} à {chosen[-1]}"
        else:
            text = ", ".join(chosen)
    return text + (" · sans ex" if no_ex else "") + (" · 1 exemplaire par carte" if singleton else "")


# --------------------------------------------------------------------------- tout-en-un


def detect(
    tournaments: pd.DataFrame,
    entries: pd.DataFrame,
    deck_cards: pd.DataFrame,
    cards: pd.DataFrame,
    sets: pd.DataFrame,
    shop: set[str],
    min_lists: int = 8,
    min_extra_lists: int = 2,
    min_records_ok: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Réserve de chaque tournoi non standard et formats détectés.

    Renvoie (colonnes à ajouter aux tournois, table des formats, cartes permises en plus par format).
    Les tournois standard gardent ``format_id = "standard"`` (sauf Dual Deck ou bilans incohérents).
    """
    t = tournaments.copy()
    usable = (t["records_ok"].fillna(0) >= min_records_ok) & ~t.get("multideck", pd.Series(False, index=t.index)).fillna(False).astype(bool)
    is_standard = t["format"].fillna("STANDARD").eq("STANDARD")
    std_ids = set(t.loc[usable & is_standard, "tournament_id"])
    matrix = ListMatrix.build(deck_cards)
    reference = t.loc[t["tournament_id"].isin(std_ids), ["tournament_id", "date"]]
    inf = Inference(cards, sets, matrix, reference, shop)

    lists_per = entries.groupby("tournament_id")["has_decklist"].sum()
    pools = []
    for row in t[usable & ~is_standard].itertuples():
        if lists_per.get(row.tournament_id, 0) < min_lists:
            continue
        pool = inf.infer(row.tournament_id, row.date)
        if pool is not None:
            pools.append(pool)

    restricted = [p for p in pools if p.restricted]
    formats = group(restricted)
    member_of = {p.tournament_id: f.format_id for f in formats for p in f.members}
    by_id = {p.tournament_id: p for p in pools}

    t["format_id"] = t["tournament_id"].map(member_of).where(~t["tournament_id"].isin(std_ids), STANDARD)
    t["pool_sets"] = t["tournament_id"].map(lambda i: "|".join(by_id[i].sets) if i in by_id else None)
    t["pool_absent"] = t["tournament_id"].map(lambda i: "|".join(by_id[i].absent) if i in by_id else None)
    t["no_ex"] = t["tournament_id"].map(lambda i: by_id[i].no_ex if i in by_id else None)
    t["singleton"] = t["tournament_id"].map(lambda i: by_id[i].singleton if i in by_id else None)
    columns = t[["tournament_id", "format_id", "pool_sets", "pool_absent", "no_ex", "singleton"]]

    info = t.set_index("tournament_id")
    rows, extra_rows = [], []
    for f in formats:
        ids = [p.tournament_id for p in f.members]
        sub = info.loc[ids]
        n_lists = sum(p.lists for p in f.members)
        extras: dict[str, int] = {}
        for p in f.members:
            for c, k in p.extras.items():
                extras[c] = extras.get(c, 0) + k
        kept = {c: k for c, k in extras.items() if k >= min_extra_lists}
        names = sub.sort_values("players", ascending=False)["name"].drop_duplicates().head(3)
        rows.append(
            {
                "format_id": f.format_id,
                "label": label(f.sets, f.no_ex, sets, f.singleton),
                "sets": "|".join(s for s in main_sets(sets)["set"] if f.sets is None or s in f.sets) if f.sets is not None else "",
                "no_ex": f.no_ex,
                "singleton": f.singleton,
                "tournaments": len(ids),
                "players": int(sub["players"].fillna(0).sum()),
                "lists": n_lists,
                "extra_cards": len(kept),
                "first_date": sub["date"].min(),
                "last_date": sub["date"].max(),
                "examples": " | ".join(names),
            }
        )
        extra_rows += [{"format_id": f.format_id, "card_id": c, "lists": k, "share": k / n_lists} for c, k in sorted(kept.items(), key=lambda x: -x[1])]
    table = pd.DataFrame(rows, columns=["format_id", "label", "sets", "no_ex", "singleton", "tournaments", "players", "lists", "extra_cards", "first_date", "last_date", "examples"])
    table = table.sort_values("players", ascending=False, ignore_index=True)
    extra = pd.DataFrame(extra_rows, columns=["format_id", "card_id", "lists", "share"])
    return columns, table, extra


def format_pool(fmt: pd.Series, extra: pd.DataFrame, shop: set[str]) -> Pool:
    """Réserve d'un format détecté (ligne de ``formats.csv``)."""
    sets = frozenset(fmt["sets"].split("|")) if isinstance(fmt["sets"], str) and fmt["sets"] else None
    cards = frozenset(extra.loc[extra["format_id"] == fmt["format_id"], "card_id"])
    return Pool(sets=sets, cards=cards, no_ex=bool(fmt["no_ex"]), always=frozenset(shop))
