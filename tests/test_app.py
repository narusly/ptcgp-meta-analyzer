"""L'interface Streamlit, exécutée sans navigateur sur le jeu de données synthétique."""

from pathlib import Path

import pytest
from synthetic import make_data, with_format, write_processed

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "ptcgp" / "app.py"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PTCGP_PROCESSED", str(write_processed(with_format(make_data()), tmp_path / "processed")))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.expander[0]  # réglages avancés présents
    at.sidebar.slider[1].set_value(3).run()  # 3 listes jouables suffisent sur ce petit jeu
    return at


def table(at, column: str, nth: int = 0):
    """Le n-ième tableau de la page qui a cette colonne."""
    return [d.value for d in at.dataframe if column in d.value.columns][nth]


def test_ranked_mode_shows_the_recommended_deck_ranking_matchups_and_lists(app):
    assert not app.exception
    assert [s.value for s in app.subheader] == ["Ton deck recommandé", "Classement", "Détail d'un deck"]
    ranking = table(app, "Victoire attendue")
    assert set(ranking["Deck"]) == {"Deck A", "Deck B", "Deck C"}
    assert ranking["Victoire attendue"].between(0, 100).all()
    top_list = table(app, "Carte")  # liste complète du deck recommandé, en haut de page
    assert "Recherches Professorales" in set(top_list["Carte"]) and top_list["Nb"].sum() == 4
    assert any("cartes" in c.value and "×" in c.value for c in app.code)  # liste à copier
    assert len(table(app, "Adversaire")) == 2  # matchups contre les deux autres decks
    labels = [t.label for t in app.tabs]
    assert labels[0] == "Liste conseillée" and labels[1] in ("Meilleure estimée", "La plus jouée") and labels[2] == "Effet des cartes"
    assert any("Limitless" in c.value for c in app.caption)  # crédits sur la page


def test_ranked_mode_with_a_single_set_filters_your_deck_only(app):
    app.sidebar.toggle[0].set_value(False).run()
    app.sidebar.multiselect[0].set_value(["C1"]).run()
    assert not app.exception
    decks = set(table(app, "Victoire attendue")["Deck"])
    assert "Deck A" not in decks and "Deck B" in decks  # Alpha n'existe qu'en B1


def test_rules_mode_uses_the_format_and_its_permitted_cards(app):
    app.sidebar.radio[0].set_value("Règles pour tout le monde").run()
    assert not app.exception
    assert app.sidebar.selectbox[0].value == "C1"
    assert any("Fusion Double (C1) seule" in c.value for c in app.caption)
    cards = app.sidebar.multiselect[1]
    assert cards.value == ["PK_4"]  # la promo d'événement jouée dans ce format
    cards.set_value([]).run()  # retirer la carte : seules les listes C sans elle comptent
    assert not app.exception
    ranking = table(app, "Victoire attendue").set_index("Deck")
    assert ranking.loc["Deck C", "Listes jouables"] == 9


def test_online_app_downloads_the_published_tables(tmp_path, monkeypatch):
    published = write_processed(with_format(make_data()), tmp_path / "branche-data" / "processed")
    monkeypatch.setenv("PTCGP_PROCESSED", str(tmp_path / "absent"))  # comme sur Streamlit Cloud
    monkeypatch.setenv("PTCGP_DATA_URL", published.as_uri())
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception and not at.error
    assert any("mise à jour chaque semaine" in c.value for c in at.caption)
