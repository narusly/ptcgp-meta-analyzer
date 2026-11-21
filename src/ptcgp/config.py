"""Chemins, adresses des sources et constantes du projet."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_CARDS = RAW / "cards"
RAW_TOURNAMENTS = RAW / "limitless"
PROCESSED = DATA / "processed"

USER_AGENT = "ptcgp-meta-analyzer/0.1 (projet open source de statistiques)"

# Base de cartes : https://github.com/flibustier/pokemon-tcg-pocket-database (licence MIT)
CARDS_REPO = "flibustier/pokemon-tcg-pocket-database"

# API publique de Limitless : https://docs.limitlesstcg.com/developer
LIMITLESS_API = "https://play.limitlesstcg.com/api"
GAME = "POCKET"

# Format standard : l'API renvoie `null` ; on écarte CUSTOM (règles maison) et NOEX (sans ex).
STANDARD_FORMATS = {None, "STANDARD"}

# Un tournoi n'est mis en cache qu'une fois terminé : on attend 48 h après son début.
SETTLE_HOURS = 48

# Codes d'extension de Limitless qui diffèrent de ceux de la base de cartes.
LIMITLESS_SET_ALIASES = {"P-A": "PROMO-A", "P-B": "PROMO-B", "P-C": "PROMO-C"}
