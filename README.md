# PTCGP Meta Analyzer

Quel deck de Pokémon TCG Pocket a le plus de chances de gagner, contre la méta du moment et avec les
extensions de son choix ? Le projet répond à partir des résultats publics des tournois Limitless :
parts de méta, matchups deck contre deck et taux de victoire attendu de chaque archétype.

> **État : étape 1 sur 4.** Le pipeline de données est en place. Viendront ensuite les statistiques
> (matchups, correction des petits échantillons), l'interface avec ses filtres, puis la mise à jour
> hebdomadaire automatique.

## Données

| Source | Contenu | Accès |
|---|---|---|
| [Limitless](https://play.limitlesstcg.com) | Tournois Pocket : decklists, archétypes, résultats de chaque match | [API publique](https://docs.limitlesstcg.com/developer), sans clé (50 requêtes / 5 min) |
| [flibustier/pokemon-tcg-pocket-database](https://github.com/flibustier/pokemon-tcg-pocket-database) | Toutes les cartes et extensions | Fichiers JSON, licence MIT |

Seuls les tournois au **format standard** et **terminés** (48 h après leur début) sont retenus. Chaque
tournoi n'est téléchargé qu'une fois, puis lu depuis le cache local (`data/raw/`, non versionné).

### Tables produites (`data/processed/`)

| Fichier | Une ligne par | Colonnes principales |
|---|---|---|
| `sets.csv` | extension | code, nom (en, fr), date de sortie |
| `prints.csv` | impression (extension + numéro) | `card_id`, rareté, PV, type, faiblesse, retraite |
| `cards.csv` | carte | `card_id`, extensions où l'obtenir (`sets`), nombre d'impressions |
| `tournaments.csv` | tournoi | date, joueurs, decklists, matchs, contrôle des bilans |
| `archetypes.csv` | archétype | identifiant et nom donnés par Limitless |
| `entries.csv.gz` | joueur inscrit | archétype, bilan, classement |
| `deck_cards.csv.gz` | carte d'une decklist | `card_id`, nombre d'exemplaires |
| `matches.csv.gz` | match | joueurs, archétypes, vainqueur, manches gagnées |

## Deux choix qui comptent

**Regrouper les impressions d'une même carte.** Une carte peut exister en plusieurs versions
(illustrations alternatives, rééditions), qui jouent toutes pareil. Le nom du fichier image contient
l'identifiant interne du jeu, commun à toutes ces versions : les 6 impressions de Mega Altaria ex
partagent `PK_011840`. Les statistiques ne suffiraient pas : les Mentali B3a-020 et C1-060 ont les mêmes
PV, type, faiblesse et retraite, mais ce sont deux cartes différentes. Résultat : 4 577 impressions,
2 478 cartes. C'est ce regroupement qui permettra de filtrer par extension : une carte reste jouable si
au moins une de ses versions vient d'une extension autorisée.

**Lire correctement les résultats.** L'API code le vainqueur d'un match par le nom du joueur, 0 pour
un nul et -1 pour une double défaite ; une ligne sans adversaire est une exemption. Ce codage a été
vérifié en recalculant le bilan de chaque joueur : il est identique à celui affiché par Limitless pour
100 % des joueurs. Ce contrôle est refait à chaque construction (colonne `records_ok`). Seuls les vrais
matchs, gagnés ou nuls, servent aux matchups.

## Utilisation

```bash
pip install -e ".[dev]"
python -m ptcgp update             # base de cartes + tournois des 90 derniers jours + tables
python -m ptcgp fetch --since 2026-10-29 --max 20
python -m ptcgp build              # reconstruit les tables depuis le cache, sans réseau
pytest -q
```

## Crédits

Données de tournois : [Limitless TCG](https://play.limitlesstcg.com). Données de cartes :
[flibustier/pokemon-tcg-pocket-database](https://github.com/flibustier/pokemon-tcg-pocket-database).
Projet non officiel, sans lien avec The Pokémon Company, Creatures, GAME FREAK ou DeNA.
