import pandas as pd

from ptcgp.archetypes import Canonicalizer, add_decks, slugify

POKEMON = ["Greninja ex", "Mega Greninja ex", "Mega Mewtwo X ex", "Mew ex", "Espeon", "Mega Altaria ex", "Farfetch'd"]


def test_slugify():
    assert slugify("Mega Mewtwo X ex") == "mega-mewtwo-x-ex"
    assert slugify("Farfetch'd") == "farfetchd"


def test_order_duplicates_share_a_key():
    canon = Canonicalizer(POKEMON)
    assert canon.key("mega-greninja-ex-c1-greninja-ex-b1") == canon.key("greninja-ex-b1-mega-greninja-ex-c1")
    # Deux Pokémon de la même extension, code d'extension commun en fin d'identifiant
    assert canon.key("mega-mewtwo-x-ex-mew-ex-c1") == canon.key("mew-ex-mega-mewtwo-x-ex-c1") == "mega-mewtwo-x-ex@c1+mew-ex@c1"


def test_versions_stay_distinct_and_unknown_ids_are_kept():
    canon = Canonicalizer(POKEMON)
    assert canon.key("mega-altaria-ex-b1-espeon-b3a") != canon.key("mega-altaria-ex-b1-espeon-c1")
    assert canon.key("unknown-mon-b1") == "unknown-mon-b1"


def test_add_decks_uses_the_most_common_name():
    entries = pd.DataFrame(
        {
            "archetype_id": ["mega-greninja-ex-c1-greninja-ex-b1"] * 3 + ["greninja-ex-b1-mega-greninja-ex-c1", None],
            "archetype": ["Mega Greninja ex Greninja ex"] * 3 + ["Greninja ex Mega Greninja ex", None],
        }
    )
    out = add_decks(entries, POKEMON)
    assert out["deck"].iloc[:4].nunique() == 1
    assert set(out["deck_name"].iloc[:4]) == {"Mega Greninja ex Greninja ex"}
    assert pd.isna(out["deck"].iloc[4])
