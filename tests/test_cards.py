import pandas as pd
import pytest

from ptcgp.cards import card_id_from_image, cards_table, clean_name, db_set_code, prints_table, sets_table

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
