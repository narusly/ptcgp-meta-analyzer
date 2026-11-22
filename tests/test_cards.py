import pandas as pd
import pytest

from ptcgp.cards import (
    _with_ex,
    card_id_from_image,
    cards_table,
    clean_name,
    db_set_code,
    derive_french,
    plausible_french,
    pokepedia_names,
    prints_table,
    sets_table,
    shop_card_ids,
)

SETS = {
    "B": [
        {"code": "B1", "releaseDate": "2025-10-30", "count": 331, "name": {"en": "Mega Rising", "fr": "Méga-Ascension"}},
        {"code": "B3a", "releaseDate": "2026-05-28", "count": 109, "name": {"en": "Paradox Drive"}},
        {"code": "B4b", "releaseDate": "2026-09-30", "count": 379, "name": {"en": "Deluxe Pack: Mega"}},
    ],
    "C": [{"code": "C1", "releaseDate": "2026-10-29", "count": 200, "name": {"en": "Dual Fusion"}}],
}


def card(set_, number, name, image, **extra):
    return {"set": set_, "number": number, "name": name, "rarity": "C", "image": image, "type": "pokemon", **extra}


CARDS = [
    card("B4b", 107, "Mega Altaria ex", "cPK_10_011840_00_MEGATYLTALISex_RR.webp", health=190),
    card("B1", 102, "Mega Altaria ex", "cPK_10_011840_00_MEGATYLTALISex_RR.webp", health=190),
    card("B3a", 107, "Mega Altaria ex", "cPK_20_011840_02_MEGATYLTALISex_SSR.webp", health=190),
    # Deux Mentali aux stats identiques mais qui sont des cartes différentes
    card("B3a", 20, "Espeon", "cPK_10_018350_00_EIFIE_U.webp", health=90),
    card("C1", 60, "Espeon", "cPK_10_022370_00_EIFIE_U.webp", health=90),
    {"set": "B1", "number": 223, "name": "Professor’s Research", "rarity": "U", "image": "cTR_10_000040_00_HAKASE_U.webp", "type": "supporter"},
]


def test_card_id_from_image_keeps_the_family_prefix():
    assert card_id_from_image("cPK_20_011840_02_MEGATYLTALISex_SSR.webp") == "PK_011840"
    # Pokémon et Dresseurs réutilisent les mêmes numéros : le préfixe les distingue
    assert card_id_from_image("cPK_10_000040_00_FUSHIGIBANAex_RR.webp") != card_id_from_image("cTR_10_000040_00_HAKASE_U.webp")


def test_card_id_from_image_rejects_unknown_names():
    with pytest.raises(ValueError):
        card_id_from_image("logo.webp")


def test_set_aliases_and_names():
    assert db_set_code("P-A") == "PROMO-A"
    assert db_set_code("B3a") == "B3a"
    assert clean_name("Professor’s Research") == "Professor's Research"


def test_cards_table_groups_reprints_and_lists_sets_in_release_order():
    sets = sets_table(SETS)
    cards = cards_table(prints_table(CARDS), sets).set_index("card_id")
    altaria = cards.loc["PK_011840"]
    assert altaria["sets"] == "B1|B3a|B4b"
    assert altaria["n_prints"] == 3
    assert altaria["first_set"] == "B1" and altaria["is_ex"]
    # Les deux Mentali restent séparés
    assert cards.loc["PK_018350", "sets"] == "B3a"
    assert cards.loc["PK_022370", "sets"] == "C1"
    assert cards.loc["TR_000040", "name"] == "Professor's Research"
    assert pd.isna(cards.loc["TR_000040", "hp"])


def test_sets_table_is_sorted_by_release():
    assert list(sets_table(SETS)["set"]) == ["B1", "B3a", "B4b", "C1"]


def test_french_names_from_tcgdex_and_the_database():
    raw = CARDS + [card("B4", 95, "Galarian Zigzagoon", "cPK_10_019000_00_ZIGZAGOON_C.webp")]
    cards_fr = [
        {"set": "B1", "number": 102, "name": "Méga-Altaria-ex"},
        {"set": "B3a", "number": 107, "name": "Feunnec"},  # traduction fausse (pas d'ex) : écartée
        {"set": "B4", "number": 95, "name": "GalarianZigzagoon"},  # anglais recollé : écarté
    ]
    tcgdex = {"B3a": {"cards": [{"localId": "020", "name": "Mentali"}]}, "A2a": {"cards": [{"localId": "023", "name": "PalkiaForme Originelle"}]}}
    prints = prints_table(raw, cards_fr, tcgdex).set_index(["set", "number"])
    assert prints.loc[("B1", 102), "name_fr"] == "Méga-Altaria-ex"
    assert pd.isna(prints.loc[("B3a", 107), "name_fr"]) and pd.isna(prints.loc[("B4", 95), "name_fr"])
    assert prints.loc[("B3a", 20), "name_fr"] == "Mentali"
    cards = cards_table(prints.reset_index(), sets_table(SETS)).set_index("card_id")
    assert cards.loc["PK_011840", "name_fr"] == "Méga-Altaria-ex"  # nom d'une autre impression de la carte
    assert cards.loc["PK_022370", "name_fr"] == "Mentali"  # même nom anglais qu'une carte traduite


def test_ex_suffix_and_derived_names():
    assert _with_ex("Chien-Pao ex", "Baojian") == "Baojian-ex"
    assert _with_ex("Mega Absol ex", "Méga-Absol-ex") == "Méga-Absol-ex"
    species = {"Greninja": "Amphinobi", "Charizard": "Dracaufeu", "Vulpix": "Goupix", "Lapras": "Lokhlass"}
    assert derive_french("Mega Greninja ex", species) == "Méga-Amphinobi-ex"
    assert derive_french("Mega Charizard Y ex", species) == "Méga-Dracaufeu Y-ex"
    assert derive_french("Alolan Vulpix", species) == "Goupix d'Alola"
    assert derive_french("Team Rocket's Lapras", species) == "Lokhlass de la Team Rocket"
    assert derive_french("Raboot", species) is None and derive_french("Greninja", species) is None
    assert not plausible_french("Mega Sceptile ex", "Feunnec") and plausible_french("Mega Absol ex", "Méga-Absol-ex")


def test_shop_cards_are_promo_a_1_to_7():
    prints = pd.DataFrame({"card_id": ["TR_1", "TR_2", "PK_9"], "set": ["PROMO-A", "A2b", "PROMO-A"], "number": [5, 111, 9]})
    assert shop_card_ids(prints) == {"TR_1"}


def test_pokepedia_names_come_from_the_card_links_of_the_set_page():
    page = {
        "title": "Fusion Double",
        "wikitext": (
            "| 123<small>/131</small>\n| [[Évoluroc (Fusion Double 123)|Évoluroc]]\n|-\n"
            "| 005<small>/131</small>\n| [[Noadkoko d'Alola-ex (Fusion Double 005)|Noadkoko <small>d'Alola</small>]]{{Symbole JCC|ex JCCP}}\n|-\n"
            "Voir aussi [[Pikachu (Promo-A 001)|Pikachu]]"  # carte d'une autre extension : ignorée
        ),
    }
    assert pokepedia_names(page) == {123: "Évoluroc", 5: "Noadkoko d'Alola-ex"}


def test_pokepedia_takes_priority_over_tcgdex_and_the_database():
    raw = CARDS + [{"set": "C1", "number": 123, "name": "Eviolite", "rarity": "C", "image": "cTR_10_001700_00_SHINKANOKISEKI_C.webp", "type": "tool"}]
    tcgdex = {"B3a": {"cards": [{"localId": "020", "name": "Mentali (TCGdex)"}]}}
    pokepedia = {
        "C1": {"title": "Fusion Double", "wikitext": "[[Évoluroc (Fusion Double 123)|Évoluroc]]"},
        "B3a": {"title": "Propulsion Paradoxe", "wikitext": "[[Mentali (Propulsion Paradoxe 020)|Mentali]]"},
    }
    prints = prints_table(raw, None, tcgdex, pokepedia).set_index(["set", "number"])
    assert prints.loc[("C1", 123), "name_fr"] == "Évoluroc"
    assert prints.loc[("B3a", 20), "name_fr"] == "Mentali"
