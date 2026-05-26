# ADIX

Un moteur Rust, une CLI et un banc d'auto-jeu pour **ADIX**, un jeu de
stratégie abstraite à deux joueurs sur un échamier 9×9, où chaque pièce
cubique présente sur sa face supérieure l'arme active (*pierre*, *feuille* ou
*ciseaux*) selon les règles de combat pierre-feuille-ciseaux.

ADIX est conçu et édité par **Échamier Games**. Le jeu, ses règles et son nom
sont leur création — achetez le jeu pour les soutenir :
<https://www.echamiergames.fr/>

Ce dépôt est une implémentation logicielle indépendante et non affiliée,
écrite pour explorer l'arbre de jeu et expérimenter avec des agents de jeu.
Les règles officielles se trouvent dans
[regle-ADIX-officielles.pdf](regle-ADIX-officielles.pdf) à la racine du dépôt ;
le moteur s'y conforme mais ne remplace pas le livret de règles.

## Compilation & exécution

Nécessite une chaîne d'outils Rust stable (édition 2024). Aucune dépendance
externe.

```sh
cargo build --release
cargo test
```

Trois binaires :

```sh
cargo run --release --bin adix                            # REPL interactif
cargo run --release --bin perft     -- 5 [--search|--tt[=mb]]
cargo run --release --bin selfplay  -- <blanc> <noir> [N] [--swap]
```

### REPL

Commandes : `help`, `board`, `moves`, `moves <case>`, `<coup>`, `undo`, `quit`.

Notation des coups :
- `e1-e2` — *déplacement* (glissement)
- `e1>n`  — *bascule* (`n`/`s`/`e`/`w`)
- `e1@l`  — *pivot* (`l` ou `r`)

Glyphes du plateau : `O` pierre, `+` feuille, `X` ciseaux, `^` abri ;
préfixe `w`/`b` pour la couleur ; `*` marque un *capitaine*.

### Auto-jeu

Agents acceptés par `selfplay` :

- `random` — coup légal uniformément aléatoire
- `ab:<profondeur>` — alpha-bêta à profondeur fixe (évaluation matérielle seule)
- `mcts:<itérations>` — MCTS UCT avec rollouts aléatoires

L'option `--swap` alterne les couleurs entre les parties pour éviter que le
résultat soit dominé par le trait.

### Perft

Nombres de nœuds verrouillés depuis la position initiale, en build release ;
le facteur de branchement tourne autour de 51 :

| coups | nœuds            | bulk               | TT (64 Mo)                     | uniques               |
|------:|-----------------:|-------------------:|-------------------------------:|----------------------:|
| 1     | 42               | <1 ms              | <1 ms                          | 41                    |
| 2     | 1 764            | <1 ms              | <1 ms                          | 1 681                 |
| 3     | 82 110           | <1 ms · 430 Mn/s   | <1 ms · 445 Mn/s               | 50 223                |
| 4     | 3 811 526        | 8 ms · 450 Mn/s    | 8 ms · 450 Mn/s                | 1 459 274             |
| 5     | 194 027 791      | 0,42 s · 465 Mn/s  | 0,26 s · 735 Mn/s              | 44 341 309            |
| 6     | 9 830 027 851    | 22 s · 450 Mn/s    | 8,6 s · 1100 Mn/s (256 Mo)     | ~1 210 025 921 (HLL)  |
| 7     | 538 293 069 289  | —                  | 4 min 51 s · 1850 Mn/s (4 Go)  | —                     |

« Uniques » compte les positions distinctes (dédupliquées par
hash Zobrist) atteignables en **exactement** N coups depuis la position
initiale — à ne pas confondre avec le perft, qui compte les feuilles de
l'arbre des coups et donc double-compte chaque transposition. Activé via
le flag opt-in `cargo run --release --bin perft -- N --unique` (exact en
HashSet jusqu'à depth 5 ; HyperLogLog à 16384 registres au-delà, erreur
attendue ~0,8 %). Le décompte exact à depth 6 demanderait ~80 Go de RAM
et reste non mesuré ici ; à depth 7 le walk HLL prend ~3 h.

Les profondeurs 0 à 3 sont figées dans [tests/perft.rs](tests/perft.rs)
côté perft, et 0 à 4 côté positions uniques. Les profondeurs 4 à 7 se
lancent à la demande via le binaire `perft`.

## Structure

- [src/lib.rs](src/lib.rs) — point d'entrée de la bibliothèque
- [src/geom.rs](src/geom.rs) — coordonnées du plateau
- [src/piece.rs](src/piece.rs) — algèbre du cube, pièces, combat PFC
- [src/board.rs](src/board.rs) — état du plateau, génération des coups, apply/unmake
- [src/zobrist.rs](src/zobrist.rs) — hachage Zobrist incrémental
- [src/perft.rs](src/perft.rs) — perft + table de transposition
- [src/agent.rs](src/agent.rs) — trait `Player`, joueurs random / alpha-bêta / MCTS
- [src/bin/adix.rs](src/bin/adix.rs) — REPL
- [src/bin/perft.rs](src/bin/perft.rs) — benchmark perft
- [src/bin/selfplay.rs](src/bin/selfplay.rs) — banc agent contre agent

Les notes d'architecture, l'algèbre du cube, le sous-ensemble de règles
encodé et les références perft sont documentés dans [CLAUDE.md](CLAUDE.md).

## État d'avancement

- Génération de tous les coups légaux, validée par perft jusqu'à la profondeur 6.
- Hachage Zobrist incrémental ; table de transposition perft fonctionnelle.
- Trois agents de base (random, alpha-bêta, MCTS). L'évaluation purement
  matérielle est le prochain chantier — voir la feuille de route dans
  [CLAUDE.md](CLAUDE.md).
- Hors périmètre : le protocole de tournoi (*j'ajuste*, annonce ADIX,
  pièce touchée pièce jouée, pendule, sanctions §11).

## Convention de vocabulaire

Le code emploie les termes français du jeu sans accents (`capitaine`,
`equipier`, `pierre`, `feuille`, `ciseaux`, `abri`, `bascule`, `pivot`,
`echamier`, `deplacement`, `clair` / `fonce`) pour rester fidèle au livret
tout en gardant des identifiants ASCII.

## Crédits & licence

ADIX — le jeu lui-même, ses règles, son nom et son identité visuelle — est la
propriété intellectuelle d'**Échamier Games**
(<https://www.echamiergames.fr/>).

Ce dépôt ne contient qu'une implémentation logicielle indépendante, écrite
pour étudier le jeu. Il n'est ni endossé par ni affilié à Échamier Games. Si
ADIX vous plaît, achetez le jeu physique chez eux.
