"""Archétypes : fusion des doublons d'ordre de Limitless.

Limitless nomme un archétype d'après ses Pokémon principaux, version comprise :
``mega-greninja-ex-c1-greninja-ex-b1``. Le même deck apparaît parfois dans l'autre ordre
(``greninja-ex-b1-mega-greninja-ex-c1``), avec des listes quasi identiques. On découpe donc
l'identifiant en Pokémon + extension, puis on trie : les deux donnent la même clé
``greninja-ex@b1+mega-greninja-ex@c1``. La version (extension) est gardée, car deux versions d'un
même Pokémon peuvent avoir des attaques différentes.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

SET_CODE = r"(?:[a-c]\d+[a-z]?|p-?[a-c]|promo-?[a-c]|sp\d+)"
_CHUNK = re.compile(rf"(.+?)-({SET_CODE})(?=-|$)")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower().replace("'", "").replace("’", "")).strip("-")


class Canonicalizer:
    """Clé d'archétype indépendante de l'ordre des Pokémon."""

    def __init__(self, pokemon_names: Iterable[str]):
        self.vocab = {slugify(n) for n in pokemon_names}

    def _segment(self, body: str) -> list[str] | None:
        """Découpe ``mega-mewtwo-x-ex-mew-ex`` en noms de Pokémon connus (le plus long d'abord)."""
        tokens = body.split("-")
        best: dict[int, list[str]] = {0: []}
        for i in range(len(tokens)):
            if i not in best:
                continue
            for j in range(len(tokens), i, -1):
                word = "-".join(tokens[i:j])
                if word in self.vocab and j not in best:
                    best[j] = best[i] + [word]
        return best.get(len(tokens))

    def key(self, archetype_id: str) -> str:
        parts, pos = [], 0
        for match in _CHUNK.finditer(archetype_id):
            if match.start() != pos:
                return archetype_id
            pokemon = self._segment(match.group(1).lstrip("-"))
            if pokemon is None:
                return archetype_id
            parts += [f"{p}@{match.group(2)}" for p in pokemon]
            pos = match.end()
        if pos != len(archetype_id) or not parts:
            return archetype_id
        return "+".join(sorted(parts))


def add_decks(entries: pd.DataFrame, pokemon_names: Iterable[str]) -> pd.DataFrame:
    """Ajoute ``deck`` (clé fusionnée) et ``deck_name`` (le nom Limitless le plus fréquent du groupe)."""
    canon = Canonicalizer(pokemon_names)
    keys = {i: canon.key(i) for i in entries["archetype_id"].dropna().unique()}
    out = entries.copy()
    out["deck"] = out["archetype_id"].map(keys)
    names = (
        out.dropna(subset=["deck"])
        .groupby(["deck", "archetype"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False, kind="stable")
        .drop_duplicates("deck")
        .set_index("deck")["archetype"]
    )
    out["deck_name"] = out["deck"].map(names)
    return out
