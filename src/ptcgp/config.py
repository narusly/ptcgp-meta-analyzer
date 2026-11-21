"""Chemins, adresses des sources et constantes du projet."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_CARDS = RAW / "cards"
RAW_TOURNAMENTS = RAW / "limitless"
# Les tests de l'interface pointent vers un jeu de données synthétique
PROCESSED = Path(os.environ.get("PTCGP_PROCESSED", DATA / "processed"))

USER_AGENT = "ptcgp-meta-analyzer/0.1 (projet open source de statistiques)"

# Base de cartes : https://github.com/flibustier/pokemon-tcg-pocket-database (licence MIT)
CARDS_REPO = "flibustier/pokemon-tcg-pocket-database"

# Noms français des cartes : https://tcgdex.dev (API publique, sans clé ; série Pocket « tcgp »)
TCGDEX_API = "https://api.tcgdex.net/v2/fr"
TCGDEX_SERIES = "tcgp"

# API publique de Limitless : https://docs.limitlesstcg.com/developer
LIMITLESS_API = "https://play.limitlesstcg.com/api"
GAME = "POCKET"

# Format standard : l'API renvoie `null`. CUSTOM (règles maison) et NOEX (sans ex) sont aussi
# téléchargés : on y cherche les formats à réserve de cartes restreinte (voir formats.py).
STANDARD_FORMATS = {None, "STANDARD"}
FETCHED_FORMATS = STANDARD_FORMATS | {"CUSTOM", "NOEX"}

# Un tournoi n'est mis en cache qu'une fois terminé : on attend 48 h après son début.
SETTLE_HOURS = 48

# Les tables ne gardent que les tournois des 120 jours précédant le plus récent.
KEEP_DAYS = 120

# Tables publiées chaque semaine par la GitHub Action sur la branche « data » ; l'app en ligne les lit
# de là quand data/processed n'existe pas (Streamlit Community Cloud ne voit que la branche main).
DATA_URL = "https://raw.githubusercontent.com/narusly/ptcgp-meta-analyzer/data/processed"  # PTCGP_DATA_URL pour la changer
PROCESSED_FILES = (
    "sets.csv", "prints.csv", "cards.csv", "tournaments.csv", "formats.csv", "format_cards.csv",
    "entries.csv.gz", "deck_cards.csv.gz", "matches.csv.gz",
)

# Codes d'extension de Limitless qui diffèrent de ceux de la base de cartes.
LIMITLESS_SET_ALIASES = {"P-A": "PROMO-A", "P-B": "PROMO-B", "P-C": "PROMO-C"}

# Dresseurs vendus en boutique contre des tickets : Potion, Vitesse +, Scrute Main, Pokédex,
# Poké Ball, Carton Rouge, Recherches Professorales. Toujours achetables, donc toujours permis.
SHOP_PRINTS = [("PROMO-A", n) for n in range(1, 8)]
