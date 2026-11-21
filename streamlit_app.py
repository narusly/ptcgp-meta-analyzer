"""Point d'entrée de Streamlit Community Cloud ; l'interface est dans src/ptcgp/app.py.

En local, préférer ``python -m ptcgp app``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ptcgp.app import main  # noqa: E402

main()
