"""Base de cartes : impressions, cartes et extensions.

Source : https://github.com/flibustier/pokemon-tcg-pocket-database (licence MIT). Noms français :
Poképédia (https://www.pokepedia.fr, licence CC BY-NC-SA 3.0), qui liste toutes les cartes de chaque
extension Pocket ; à défaut TCGdex (https://tcgdex.dev, jusqu'à B2a), puis les traductions de la base.

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
import urllib.parse
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import CARDS_REPO, LIMITLESS_SET_ALIASES, POKEPEDIA_API, SHOP_PRINTS, TCGDEX_API, TCGDEX_SERIES
from .http import HttpClient

IMAGE_ID = re.compile(r"^c([A-Z]+)_\d+_(\d{6})_\d{2}_")
CARD_FILES = ("cards.extra.json", "sets.json")
FRENCH_FILE = "cards.fr.json"
TCGDEX_DIR = "tcgdex_fr"
POKEPEDIA_DIR = "pokepedia"
# Lien vers la page d'une carte dans la liste d'une extension : [[Évoluroc (Fusion Double 123)|…]]
_POKEPEDIA_LINK = re.compile(r"\[\[([^\[\]|]+?) \(([^()\[\]|]+?) (\d{3})\)\|")


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
    for name in CARD_FILES + (FRENCH_FILE,):
        raw = client.get_bytes(f"https://raw.githubusercontent.com/{CARDS_REPO}/{ref}/dist/{name}")
        (dest / name).write_bytes(raw)
        data[name] = json.loads(raw.decode("utf-8"))
    (dest / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    try:
        data[TCGDEX_DIR] = download_tcgdex(client, dest / TCGDEX_DIR)
    except Exception as err:  # noms français facultatifs : on garde ceux déjà en cache
        log(f"Noms français de TCGdex indisponibles ({err}) : cache précédent conservé")
    alt = {db_set_code(k): v.get("name") for k, v in (data.get(TCGDEX_DIR) or {}).items()}  # autre nom français (TCGdex)
    data[POKEPEDIA_DIR] = download_pokepedia(HttpClient(min_interval=1.0, log=log), dest / POKEPEDIA_DIR, data["sets.json"], log, alt)
    return data


def download_pokepedia(client: HttpClient, dest: Path, sets_json: dict, log: Callable[[str], None] = print,
                       alt_titles: dict[str, str] | None = None) -> dict[str, dict]:
    """Page Poképédia de chaque extension (une requête par seconde) : son tableau « Liste des cartes »
    donne le nom français officiel de chaque carte. On essaie le nom français de la base, puis celui de
    TCGdex (« Gardiens Astraux » pour A3). Une page introuvable garde sa version en cache."""
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for code, title in ((s["code"], (s.get("name") or {}).get("fr")) for group in sets_json.values() for s in group):
        titles = list(dict.fromkeys(t.replace("’", "'") for t in (title, (alt_titles or {}).get(code)) if t))
        data = {}
        for candidate in titles:
            query = {"action": "parse", "page": candidate, "prop": "wikitext", "redirects": 1, "format": "json"}
            try:
                data = client.get_json(f"{POKEPEDIA_API}?{urllib.parse.urlencode(query)}")
            except Exception as err:
                log(f"Poképédia indisponible pour {code} ({err})")
                data = {}
            if "parse" in data:
                break
        if "parse" not in data:
            if titles:
                log(f"Pas de page Poképédia pour {code} ({', '.join(titles)})")
            continue
        page = {"title": data["parse"]["title"], "wikitext": data["parse"]["wikitext"]["*"]}
        (dest / f"{code}.json").write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
        out[code] = page
    return out


def pokepedia_names(page: dict) -> dict[int, str]:
    """Numéro → nom français des cartes de la liste d'une extension (liens vers les pages des cartes)."""
    return {int(n): clean_name(name) for name, set_title, n in _POKEPEDIA_LINK.findall(page.get("wikitext", ""))
            if set_title == page.get("title")}


def download_tcgdex(client: HttpClient, dest: Path) -> dict[str, dict]:
    """Noms français de chaque extension Pocket de TCGdex (une requête par extension)."""
    dest.mkdir(parents=True, exist_ok=True)
    series = client.get_json(f"{TCGDEX_API}/series/{TCGDEX_SERIES}")
    out = {}
    for s in series.get("sets", []):
        data = client.get_json(f"{TCGDEX_API}/sets/{s['id']}")
        (dest / f"{s['id']}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        out[s["id"]] = data
    return out


def load(dest: Path) -> dict:
    data = {name: json.loads((dest / name).read_text(encoding="utf-8")) for name in CARD_FILES}
    if (dest / FRENCH_FILE).exists():
        data[FRENCH_FILE] = json.loads((dest / FRENCH_FILE).read_text(encoding="utf-8"))
    for folder in (TCGDEX_DIR, POKEPEDIA_DIR):
        if (dest / folder).exists():
            data[folder] = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((dest / folder).glob("*.json"))}
    return data


def _with_ex(english: str, french: str) -> str:
    """TCGdex écrit « Baojian » pour Chien-Pao ex : on remet le suffixe, comme la base (« Méga-Absol-ex »)."""
    if english.endswith(" ex") and not french.lower().endswith("ex"):
        return f"{french}-ex"
    return french


def plausible_french(english: str, french: str) -> bool:
    """Écarte les « traductions » de la base qui n'en sont pas : nom anglais recollé
    (« GalarianMeowth »), ou carte ex traduite par un autre Pokémon (Mega Sceptile ex → « Feunnec »)."""
    if not french or french == english or french.replace(" ", "") == english.replace(" ", ""):
        return False
    return not (english.endswith(" ex") and not french.lower().endswith("ex"))


REGIONS = {"Alolan": "d'Alola", "Galarian": "de Galar", "Paldean": "de Paldea", "Hisuian": "de Hisui"}
_FORM = re.compile(r"^(?P<mega>Mega )?(?P<rocket>Team Rocket's )?(?:(?P<region>Alolan|Galarian|Paldean|Hisuian) )?(?P<base>.+?)(?P<xy> [XY])?(?P<ex> ex)?$")


def derive_french(english: str, species: dict[str, str]) -> str | None:
    """Nom français reconstruit à partir de l'espèce : « Mega Greninja ex » → « Méga-Amphinobi-ex »,
    « Alolan Vulpix » → « Goupix d'Alola ». None si l'espèce n'a pas de nom français connu."""
    m = _FORM.match(english)
    if not m or m.group("base") not in species or m.group("base") == english:
        return None
    name = ("Méga-" if m.group("mega") else "") + species[m.group("base")] + (m.group("xy") or "")
    if m.group("region"):
        name += f" {REGIONS[m.group('region')]}"
    if m.group("rocket"):
        name += " de la Team Rocket"
    return name + ("-ex" if m.group("ex") else "")


def french_print_names(prints: pd.DataFrame, cards_fr: list[dict] | None, tcgdex: dict[str, dict] | None,
                       pokepedia: dict[str, dict] | None = None) -> pd.Series:
    """Nom français de chaque impression quand une source le donne (sinon NaN), aligné sur ``prints``.

    Ordre de priorité : Poképédia, TCGdex, puis les traductions de la base qui passent ``plausible_french``.
    """
    names: dict[tuple[str, int], str] = {}
    english = dict(zip(zip(prints["set"], prints["number"]), prints["name"]))
    for c in cards_fr or []:
        key = (c["set"], int(c["number"]))
        name = clean_name(c.get("name", ""))
        if key in english and plausible_french(english[key], name):
            names[key] = name
    for set_id, data in (tcgdex or {}).items():
        code = db_set_code(set_id)
        for c in data.get("cards", []):
            local = str(c.get("localId", ""))
            key = (code, int(local)) if local.isdigit() else None
            if key in english and c.get("name"):
                # TCGdex écrit « PalkiaForme Originelle » : on rétablit l'espace manquant
                name = re.sub(r"([a-zàâçéèêëîïôûùüÿœ])([A-Z])", r"\1 \2", clean_name(c["name"]))
                names[key] = _with_ex(english[key], name)
    for code, page in (pokepedia or {}).items():
        for number, name in pokepedia_names(page).items():
            if (code, number) in english:
                names[(code, number)] = _with_ex(english[(code, number)], name)
    return pd.Series([names.get(k) for k in zip(prints["set"], prints["number"])], index=prints.index, dtype=object)


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


def prints_table(cards_json: list[dict], cards_fr: list[dict] | None = None, tcgdex: dict[str, dict] | None = None,
                 pokepedia: dict[str, dict] | None = None) -> pd.DataFrame:
    """Une ligne par impression (extension + numéro), avec l'identifiant de la carte et son nom français."""
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
    df.insert(df.columns.get_loc("name") + 1, "name_fr", french_print_names(df, cards_fr, tcgdex, pokepedia))
    return df


def shop_card_ids(prints: pd.DataFrame) -> set[str]:
    """Dresseurs de boutique (voir ``config.SHOP_PRINTS``), toujours permis."""
    keys = set(SHOP_PRINTS)
    return {cid for cid, key in zip(prints["card_id"], zip(prints["set"], prints["number"])) if key in keys}


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
    if "name_fr" in prints:
        known = ordered.dropna(subset=["name_fr"])
        by_card = known.drop_duplicates("card_id").set_index("card_id")["name_fr"]
        by_name = known.groupby("name")["name_fr"].agg(lambda s: s.mode().iloc[0])
        name_fr = by_card.reindex(out.index).fillna(out["name"].map(by_name))
        derived = out["name"].map(lambda n: derive_french(n, by_name.to_dict()))
        out.insert(1, "name_fr", name_fr.fillna(derived))
    out["is_ex"] = out["name"].str.endswith(" ex")
    out["sets"] = sets_per_card
    out["n_prints"] = ordered.groupby("card_id").size()
    out["first_set"] = first["set"]
    out["first_release"] = first["set"].map(release)
    return out.reset_index()
