# PTCGP Meta Analyzer

Quel deck de Pokémon TCG Pocket a le plus de chances de gagner, contre la méta du moment et avec les
extensions de son choix ? Le projet répond à partir des résultats publics des tournois Limitless :
parts de méta, matchups deck contre deck et taux de victoire attendu de chaque archétype.

> **État : étape 2 sur 4.** Données et statistiques en place (ligne de commande). Viendront ensuite
> l'interface avec ses filtres, puis la mise en ligne et la mise à jour hebdomadaire automatique.

## En bref (méta du 29 octobre au 18 novembre 2026, 87 tournois, 25 953 matchs)

| Deck | Victoire attendue en classé [IC 90 %] | P(meilleur deck) | Popularité |
|---|---|---|---|
| Mega Altaria ex Espeon | 56,4 % [54,8 ; 57,9] | 38 % | 3,4 % |
| Magnezone Miraidon ex | 56,1 % [54,6 ; 57,5] | 22 % | 4,1 % |
| Mega Altaria ex Greninja | 55,8 % [54,4 ; 57,3] | 14 % | 3,9 % |
| Hoopa ex Mega Absol ex | 55,5 % [54,1 ; 56,9] | 7 % | 4,7 % |
| Suicune ex Baxcalibur | 55,1 % [53,8 ; 56,4] | 3 % | 5,3 % |

- **Aucun deck ne domine.** Les cinq premiers se tiennent en 1,3 point, et le premier n'a que 38 % de
  chances d'être vraiment le meilleur. Le deck le plus joué (Mega Charizard Y ex Entei ex, 9,6 % des
  joueurs) ne figure pas dans ce haut de tableau.
- **Les matchups comptent au-delà de la force globale.** Sur des tournois que le modèle n'avait pas
  vus, le lissage vers Bradley-Terry prédit mieux que Bradley-Terry seul, que le taux brut et que le
  pile ou face (gains significatifs, voir [Validation](#validation)).

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

## Méthode statistique

L'objectif est le **classé** : des parties en une manche contre des adversaires au hasard.

1. **Période et tournois.** Par défaut, depuis la sortie de la dernière extension. Les tournois dont
   les bilans recalculés ne collent pas (moins de 95 %) sont écartés : ce sont des tournois par
   équipes que l'API classe en format standard.
2. **Archétypes.** Limitless écrit parfois le même deck dans deux ordres (« Mega Greninja ex Greninja
   ex » et « Greninja ex Mega Greninja ex », aux listes quasi identiques). L'identifiant est découpé en
   Pokémon + version puis trié, ce qui fusionne 112 groupes de doublons. Les decks joués par moins de
   0,5 % des joueurs sont regroupés en « Autres decks ».
3. **Unité d'observation : la manche.** Chaque manche d'un match en deux manches gagnantes compte ; un
   match sans détail des manches vaut une manche (vérifié : ce ne sont pas des forfaits) ; un nul vaut
   une demi-victoire.
4. **Force globale : modèle de Bradley-Terry.** P(A bat B) = πA / (πA + πB), estimé par l'algorithme MM
   de Hunter (2004).
5. **Matchups lissés par Bayes empirique.** Un duel A contre B observé sur peu de manches est tiré vers
   la prédiction de Bradley-Terry p0, avec un poids κ en manches fictives :
   P(A bat B) ~ Bêta(victoires + κ·p0, défaites + κ·(1 − p0)). κ maximise la vraisemblance
   bêta-binomiale de tous les duels observés (κ = 33 sur la période actuelle).
6. **Taux de victoire attendu en classé** : moyenne des matchups pondérée par la popularité des decks
   adverses (miroir à 50 %).
7. **Incertitude** : 4 000 tirages dans les lois a posteriori (bêta pour les matchups, Dirichlet pour la
   méta) donnent un intervalle à 90 % et la probabilité que chaque deck soit le meilleur.

**Filtre par extension.** Il ne s'applique qu'à son propre deck : les adversaires gardent la vraie
méta. Une liste est jouable si chacune de ses cartes existe dans une extension cochée (les promos sont
autorisées par défaut). Pour un deck candidat, seules ses listes jouables comptent, et ses autres listes
servent d'a priori : a = victoires jouables + κ·q, où q vient des autres listes, elle-même lissée vers
Bradley-Terry. Sans filtre, on retrouve exactement le modèle complet (testé). Comme les decks de méta
mélangent beaucoup d'extensions, une **tolérance** accepte les listes auxquelles il manque quelques
exemplaires, en signalant les cartes à trouver ou à remplacer.

### Validation

Modèles appris sur les tournois du 29 octobre au 11 novembre, testés sur ceux du 12 au 18 novembre
(19 tournois, 5 398 manches). Plus bas = meilleur.

| Modèle | Perte logarithmique | Score de Brier |
|---|---|---|
| Pile ou face (50 %) | 0,6931 | 0,2500 |
| Taux brut observé | 0,7010 | 0,2505 |
| Bradley-Terry seul | 0,6922 | 0,2495 |
| **Bradley-Terry + lissage bayésien** | **0,6872** | **0,2470** |

Écart de perte logarithmique par manche du modèle retenu, avec un intervalle de confiance à 95 % par
bootstrap sur les tournois : −0,0060 [−0,0122 ; −0,0004] contre le pile ou face, −0,0138 [−0,0198 ;
−0,0071] contre le taux brut, −0,0050 [−0,0086 ; −0,0014] contre Bradley-Terry seul. Le taux brut fait
pire que le hasard : il colle au bruit des petits échantillons. Les gains restent modestes en valeur
absolue, car une manche de Pocket garde une grosse part de hasard (pioche, pile ou face).

### Limites

- La méta des tournois sert de substitut à celle du classé, qui n'est pas publique.
- Avec la tolérance, le taux affiché est celui des listes d'origine : remplacer des cartes peut le
  faire baisser.
- Certaines promos sont des récompenses d'événements passés, introuvables aujourd'hui.

## Utilisation

```bash
pip install -e ".[dev]"
python -m ptcgp update             # base de cartes + tournois des 90 derniers jours + tables
python -m ptcgp build              # reconstruit les tables depuis le cache, sans réseau
python -m ptcgp stats              # classement des decks pour le classé, et la liste type du premier
python -m ptcgp stats --sets C1,B4b --tolerance 2   # seulement Dual Fusion et le Deluxe Pack
python -m ptcgp stats --validate   # test sur les tournois les plus récents
pytest -q
```

## Crédits

Données de tournois : [Limitless TCG](https://play.limitlesstcg.com). Données de cartes :
[flibustier/pokemon-tcg-pocket-database](https://github.com/flibustier/pokemon-tcg-pocket-database).
Projet non officiel, sans lien avec The Pokémon Company, Creatures, GAME FREAK ou DeNA.
