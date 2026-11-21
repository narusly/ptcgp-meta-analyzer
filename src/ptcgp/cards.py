"""Base de cartes : impressions, cartes et extensions.

Source : https://github.com/flibustier/pokemon-tcg-pocket-database (licence MIT).

Une même carte peut être imprimée plusieurs fois (illustrations alternatives, rééditions dans un autre
pack). Toutes ces impressions jouent pareil. Pour les regrouper, on utilise l'identifiant interne du
jeu, présent dans le nom du fichier image et commun à toutes les impressions d'une carte :
``cPK_10_011840_00_MEGATYLTALISex_RR.webp`` et ``cPK_20_011840_02_MEGATYLTALISex_SSR.webp`` sont deux
impressions de Mega Altaria ex (``PK_011840``). Les stats ne suffiraient pas : les Mentali B3a-020 et
C1-060 ont les mêmes PV, type, faiblesse et retraite, mais ce sont deux cartes différentes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import CARDS_REPO, LIMITLESS_SET_ALIASES
from .http import HttpClient

IMAGE_ID = re.compile(r"^c([A-Z]+)_\d+_(\d{6})_\d{2}_")
CARD_FILES = ("cards.extra.json", "sets.json")


def card_id_from_image(image: str) -> str:
    """Identifiant commun à toutes les impressions d'une carte : ``PK_011840``, ``TR_000040``...

    Le préfixe (PK pour les Pokémon, TR pour les Dresseurs) fait partie de l'identifiant : les deux
    familles réutilisent les mêmes numéros.
    """
    match = IMAGE_ID.match(image)
    if not match:
        raise ValueError(f"Nom d'image inattendu : {image!r}")
    return f"{match.group(1)}_{match.group(2)}"


def db_set_code(limitless_set: str) -> str:
    """Code d'extension de Limitless (``P-A``) vers celui de la base de cartes (``PROMO-A``)."""
    return LIMITLESS_SET_ALIASES.get(limitless_set, limitless_set)


def clean_name(name: str) -> str:
    """Harmonise les apostrophes : la base écrit « Professor’s Research », Limitless « Professor's »."""
    return name.replace("’", "'").strip()


def download(client: HttpClient, dest: Path, log: Callable[[str], None] = print) -> dict:
    """Télécharge la base au dernier commit de ``main`` et note ce commit dans ``source.json``."""
    dest.mkdir(parents=True, exist_ok=True)
    source = {"repo": CARDS_REPO, "commit": None, "date": None}
    try:
        commit = client.get_json(f"https://api.github.com/repos/{CARDS_REPO}/commits/main")
        source.update(commit=commit["sha"], date=commit["commit"]["committer"]["date"])
    except Exception as err:  # l'API GitHub limite les appels anonymes : on se rabat sur `main`
        log(f"Commit de la base introuvable ({err}) : téléchargement depuis main")
    ref = source["commit"] or "main"
    data = {}
    for name in CARD_FILES:
        raw = client.get_bytes(f"https://raw.githubusercontent.com/{CARDS_REPO}/{ref}/dist/{name}")
        (dest / name).write_bytes(raw)
        data[name] = json.loads(raw.decode("utf-8"))
    (dest / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    return data


def load(dest: Path) -> dict:
    return {name: json.loads((dest / name).read_text(encoding="utf-8")) for name in CARD_FILES}


def sets_table(sets_json: dict) -> pd.DataFrame:
    """Une ligne par extension, dans l'ordre de sortie."""
    rows = []
    for series, sets in sets_json.items():
        for s in sets:
            names = s.get("name") or {}
            rows.append(
                {
                    "set": s["code"],
                    "series": series,
                    "name": names.get("en", s["code"]),
                    "name_fr": names.get("fr", ""),
                    "release_date": s.get("releaseDate"),
                    "count": s.get("count"),
                }
            )
    return pd.DataFrame(rows).sort_values(["release_date", "set"], ignore_index=True)


def prints_table(cards_json: list[dict]) -> pd.DataFrame:
    """Une ligne par impression (extension + numéro), avec l'identifiant de la carte."""
    rows = [
        {
            "card_id": card_id_from_image(c["image"]),
            "set": c["set"],
            "number": int(c["number"]),
            "name": clean_name(c["name"]),
            "rarity": c.get("rarity"),
            "category": c.get("type"),
            "element": c.get("element"),
            "stage": c.get("stage"),
            "hp": c.get("health"),
            "retreat_cost": c.get("retreatCost"),
            "weakness": c.get("weakness"),
            "evolves_from": clean_name(c["evolvesFrom"]) if c.get("evolvesFrom") else None,
            "packs": "|".join(c.get("packs") or []),
        }
        for c in cards_json
    ]
    df = pd.DataFrame(rows)
    for col in ("hp", "retreat_cost"):
        df[col] = df[col].astype("Int64")
    return df


def cards_table(prints: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par carte, avec toutes les extensions où l'on peut l'obtenir (dans l'ordre de sortie).

    C'est cette liste qui sert au filtre par extension : une carte reste jouable si au moins une de
    ses extensions est autorisée.
    """
    order = {code: i for i, code in enumerate(sets["set"])}
    release = dict(zip(sets["set"], sets["release_date"]))
    ordered = prints.assign(_order=prints["set"].map(order).fillna(len(order))).sort_values(["_order", "number"])
    first = ordered.drop_duplicates("card_id").set_index("card_id")
    sets_per_card = ordered.groupby("card_id", sort=False)["set"].agg(lambda s: "|".join(dict.fromkeys(s)))
    out = first[["name", "category", "element", "stage", "hp", "retreat_cost", "weakness", "evolves_from"]].copy()
    out["is_ex"] = out["name"].str.endswith(" ex")
    out["sets"] = sets_per_card
    out["n_prints"] = ordered.groupby("card_id").size()
    out["first_set"] = first["set"]
    out["first_release"] = first["set"].map(release)
    return out.reset_index()
