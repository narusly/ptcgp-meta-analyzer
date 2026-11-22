"""Figure du README et du portfolio : ``python -m ptcgp figures`` → reports/figures/classement.png.

Deux classements côte à côte, avec leur intervalle à 90 % : le classé (méta standard) et le format
« Fusion Double seule », où les règles changent pour tout le monde. Même style que les autres projets
du portfolio (palette validée, encre et grille discrètes).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from . import stats

SERIES = "#2a78d6"  # emplacement 1 de la palette validée
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "ytick.left": False,
    })


def ranking_panel(ax, ranking: pd.DataFrame, title: str, subtitle: str, top: int) -> None:
    """Points (victoire attendue) et segments (intervalle à 90 %), le meilleur en haut."""
    r = ranking.head(top).iloc[::-1]
    y = range(len(r))
    ax.hlines(y, r["low"] * 100, r["high"] * 100, color=SERIES, linewidth=2)
    ax.scatter(r["win_rate"] * 100, y, s=42, color=SERIES, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(50, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks(list(y), [f"{n}  " for n in r["name_fr"]])
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Victoire attendue par manche (%)")
    ax.set_title(title, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9)
    best = r.iloc[-1]
    label = f"P(meilleur) {best['p_best'] * 100:.0f} %"
    ax.annotate(label, (best["win_rate"] * 100, len(r) - 1), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", color=INK_2, fontsize=9)
    ax.set_ylim(-0.6, len(r) - 0.1)  # place pour l'annotation, sous le sous-titre


def ranking_figure(data: stats.Data, path: Path, fmt: str = "C1", top: int = 8) -> Path:
    apply_style()
    standard = stats.analyze(data)
    special = stats.analyze(data, format_id=fmt)
    label = data.formats.set_index("format_id").loc[fmt, "label"]
    # Panneaux l'un sous l'autre : la figure reste lisible sur une page étroite (portfolio, téléphone)
    fig, (left, right) = plt.subplots(2, 1, figsize=(7.5, 9.2), gridspec_kw={"hspace": 0.55})
    n_std, n_fmt = len(standard.data.tournaments), len(special.data.tournaments)
    players = int(special.data.tournaments["players"].sum())
    ranking_panel(left, standard.ranking, "Classé : méta des tournois standard",
                  f"{n_std} tournois depuis la dernière extension · intervalle à 90 %", top)
    ranking_panel(right, special.ranking, f"Règles pour tout le monde : {label}",
                  f"{n_fmt} tournoi{'s' if n_fmt > 1 else ''}, {players} joueurs · une tout autre méta", top)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
