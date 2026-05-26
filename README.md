# ADIX

A Rust engine, CLI, and self-play harness for **ADIX**, a 2-player abstract
strategy game on a 9×9 board with cubic pieces whose top face is the active
arme (*pierre / feuille / ciseaux*) under rock-paper-scissors combat rules.

ADIX is designed and published by **Échamier Games**. The game, its rules, and
its name are their work — please buy a copy and support them:
<https://www.echamiergames.fr/>

This repository is an independent, unaffiliated implementation built to
explore the game tree and experiment with playing agents. The canonical rules
live in [regle-ADIX-officielles.pdf](regle-ADIX-officielles.pdf) at the repo
root; the engine follows them but is not a substitute for the rulebook.

## Build & run

Requires a stable Rust toolchain (edition 2024). No external crates.

```sh
cargo build --release
cargo test
```

Three binaries:

```sh
cargo run --release --bin adix                            # interactive REPL
cargo run --release --bin perft     -- 5 [--search|--tt[=mb]]
cargo run --release --bin selfplay  -- <white> <black> [N] [--swap]
```

### REPL

Commands: `help`, `board`, `moves`, `moves <sq>`, `<move>`, `undo`, `quit`.

Move notation:
- `e1-e2` — *déplacement* (slide)
- `e1>n`  — *bascule* (tumble: `n`/`s`/`e`/`w`)
- `e1@l`  — *pivot* (`l` or `r`)

Board glyphs: `O` pierre, `+` feuille, `X` ciseaux, `^` abri; `w`/`b` color
prefix; `*` marks a *capitaine*.

### Self-play

Agent specs accepted by `selfplay`:

- `random` — uniform random legal move
- `ab:<depth>` — fixed-depth alpha-beta (material-only eval)
- `mcts:<iterations>` — UCT MCTS with random rollouts

`--swap` alternates colors between games so results aren't biased by who moves
first.

### Perft

Locked node counts from the initial position, release build, branching factor
~51:

| depth | nodes        | bulk  | search | TT (64 MB) |
|------:|-------------:|------:|-------:|-----------:|
| 3     | 82 110       | <1 ms | 2 ms   | <1 ms      |
| 4     | 3 811 526    | 8 ms  | 95 ms  | 8 ms       |
| 5     | 194 027 791  | 0.42 s| 4.8 s  | 0.26 s     |
| 6     | 9 830 027 851| 22 s  | —      | 8.6 s (256 MB) |

Depths 0–3 are pinned in [tests/perft.rs](tests/perft.rs); depths 4–6 run on
demand via the `perft` binary.

## Layout

- [src/lib.rs](src/lib.rs) — library entry point
- [src/geom.rs](src/geom.rs) — board coordinates
- [src/piece.rs](src/piece.rs) — cube algebra, pieces, RPS combat
- [src/board.rs](src/board.rs) — board state, move generation, apply/unmake
- [src/zobrist.rs](src/zobrist.rs) — incremental Zobrist hashing
- [src/perft.rs](src/perft.rs) — perft + transposition table
- [src/agent.rs](src/agent.rs) — `Player` trait, random / alpha-beta / MCTS
- [src/bin/adix.rs](src/bin/adix.rs) — REPL
- [src/bin/perft.rs](src/bin/perft.rs) — perft benchmark
- [src/bin/selfplay.rs](src/bin/selfplay.rs) — agent-vs-agent harness

Architectural notes, the cube algebra, the rules subset that's encoded, and
the perft baselines are documented in [CLAUDE.md](CLAUDE.md).

## Status

- Full legal move generation, validated by perft to depth 6.
- Incremental Zobrist hashing; perft TT working.
- Three baseline agents (random, alpha-beta, MCTS). The material-only eval is
  the next thing to improve — see the roadmap in [CLAUDE.md](CLAUDE.md).
- Out of scope: tournament protocol (*j'ajuste*, ADIX announcement, touch-move,
  clock, §11 sanctions).

## Vocabulary

The code uses French game-domain terms without accents (`capitaine`,
`equipier`, `pierre`, `feuille`, `ciseaux`, `abri`, `bascule`, `pivot`,
`echamier`, `deplacement`, `clair` / `fonce`) to stay faithful to the
rulebook while keeping identifiers ASCII.

## Credit & licence

ADIX — the game itself, its rules, its name, and its visual identity — is the
intellectual property of **Échamier Games** (<https://www.echamiergames.fr/>).

This repository contains only an independent software implementation written
to study the game. It is not endorsed by or affiliated with Échamier Games. If
you enjoy ADIX, buy the physical game from them.
