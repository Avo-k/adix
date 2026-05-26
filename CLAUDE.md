# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust engine + CLI for **ADIX** (Echamier Games) — a 2-player abstract strategy
game on a 9×9 board with cubic pieces whose top face is the active arme
(pierre/feuille/ciseaux) under RPS combat rules. The repo exists to explore
strategies and the game tree; there is no AI yet.

The canonical rules are in `regle-ADIX-officielles.pdf` at the repo root. If
something in the engine looks wrong, **the PDF is the source of truth** — not
prior conversation, not memory.

## Commands

```sh
cargo build              # debug build of lib + bin
cargo test               # all 27 tests (unit + integration)
cargo test <name>        # single test by substring, e.g. cargo test capture_blocked
cargo test --test rules  # one integration file (tests/rules.rs)
cargo run                # launch the REPL on the initial position
cargo build --release    # optimized build, useful before tree-search work
```

The REPL accepts: `help`, `board`, `moves`, `moves <sq>`, `<move>`, `undo`,
`quit`. Move notation: `e1-e2` (deplacement), `e1>n` (bascule), `e1@l|r` (pivot).
Board glyphs: `O` pierre, `+` feuille, `X` ciseaux, `^` abri; `w`/`b` color
prefix; `*` marks a capitaine.

## Vocabulary convention (load-bearing)

**Code uses French terms without accents** for game-domain types: `capitaine`,
`equipier`, `pierre`, `feuille`, `ciseaux`, `abri`, `bascule`, `pivot`,
`echamier`, `deplacement`, `clair`/`fonce` (white/black). Do not anglicize
them. Adding accents (e.g. `équipier`) will break identifiers.

## Architecture

Five modules, kept deliberately small and dependency-free (no external crates):

- `geom` — `Pos { file, rank }` with file 0–8 = a–i, rank 0–8 = 1–9. `Pos::is_dark`
  uses `(file + rank) % 2 == 0`, so a1 (the dark corner) is dark.
- `piece` — `Arme`, `Face` (`Arme | Abri`), `Cube` (6 oriented faces),
  `Piece` carrying `last_kind: Option<MoveKind>` + `streak: u8` for the 3× rule.
- `moves` — `Move` enum + `slide_dir` helper that classifies a `(from, to)` pair.
- `board` — `Board` holds the 9×9 grid, side to move, `plies_since_progress`
  draw counter, captured list. `legal_moves`, `apply`, `outcome` live here.
- `notation` — parsing + ASCII rendering.

The library is in `src/lib.rs`; the REPL is `src/bin/adix.rs` (separate target).

### Cube algebra — the only piece of math that matters

Axes: `+y = N`, `+x = E`, `+z = up`. The cube struct stores `{top, bottom,
north, south, east, west}`.

- `bascule(N)` tumbles forward over the north edge → `top := old.south,
  bottom := old.north, north := old.top, south := old.bottom`; east/west
  unchanged. The other three directions are derived analogously.
- `pivot(Right)` rotates clockwise viewed from above → `north := old.west,
  east := old.north, south := old.east, west := old.south`; top/bottom unchanged.

These were derived from first principles and unit-tested (`bascule(d)` ×4 and
`pivot(r)` ×4 must be identity, opposite-direction pairs cancel). **Do not edit
this algebra without re-running the cube tests and re-checking starting
orientation effects** — a sign flip silently desyncs the whole engine.

### Rules encoded (where to look first)

- Starting position is hard-coded in `Board::initial()` per the PDF (white
  capitaine e1; equipiers c1 g1 b2 d2 f2 h2 c3 e3 g3; black mirrored). It
  never varies. PDF §11-1 says an incorrect starting layout voids the game.
- Capture is RPS with `Arme::beats` in `piece.rs`. Equal armes is illegal
  (cannot land on a piece you can't beat). Capture only happens via
  deplacement (§8-4); bascule never captures. A piece showing `Abri` on top
  is unattackable (§8-5).
- Sliding stops at the first non-empty square. No jumping (§7-2-7 / §8-2-2).
- Capitaine slides exactly 1 square; equipier slides any distance.
- The **3× rule** (§7-3-3): a piece's `streak` increments while its
  `last_kind` equals the new move's kind (direction-irrelevant, per user
  decision). Other pieces moving in between do **not** reset this piece's
  streak — it's a per-piece counter on that piece's own move history. When
  `streak` hits 3, the piece self-removes (`commit_move` pushes it to
  `captured` instead of placing it).
- Draw counter resets only on a deplacement; basculsand pivots increment it.
  Limit is `DRAW_PLY_LIMIT = 30` (§10-2-1).

### Intentionally out of scope

The engine ignores tournament protocol: no `j'ajuste`, no `ADIX`
pre-capture announcement, no touch-move, no clock. Irregularity sanctions
(§11) are not modelled — `apply()` simply returns `Err(IllegalMove::…)` and
leaves the board untouched. Add these only if the user explicitly asks.

## Test layout

- Unit tests live in `#[cfg(test)] mod tests` blocks inside each module
  (currently `geom` and `piece`).
- Integration tests in `tests/`:
  - `initial.rs` — starting position structure.
  - `moves_basic.rs` — reproduces all three capture scenarios from the PDF's
    Annexe 3 as fixtures. When adding move-generation tests, do the same:
    build positions via `Board::empty_for_test()` + `force_place` rather than
    massaging the initial position.
  - `rules.rs` — 3× removal, draw counter, win conditions.

`Board::empty_for_test`, `force_place`, and `set_side_to_move` are
test-affordance APIs on the public surface; keep them if you need to write
new fixtures. They're safe in regular code too but the names signal intent.

## Perft baseline

From `Board::initial()`, current move generator (clone-on-make, `Vec<Move>` per
node, release build):

| depth | nodes        | time   | M nodes/s |
|------:|-------------:|-------:|----------:|
| 1     | 42           | <1 ms  | —         |
| 2     | 1 764        | 1 ms   | 2.5       |
| 3     | 82 110       | 31 ms  | 2.7       |
| 4     | 3 811 526    | 1.5 s  | 2.5       |
| 5     | 194 027 791  | 107 s  | 1.8       |

Branching factor settles around ~51. Depths 0–3 are locked in
[tests/perft.rs](tests/perft.rs); depths 4–5 are run on demand via
`./target/release/perft <max_depth> [divide_at]`. **If you touch move
generation, re-run perft 4 — a single off-by-one shows up loudly here.**

## Where this is going (big next steps)

In rough priority order — the first two are prerequisites for anything else:

1. **Make-unmake instead of clone-on-make.** Replace `let mut child =
   board.clone(); child.apply(mv)` with `board.apply(mv); …; board.undo(mv)`.
   Saves ~95 % of the per-node allocation, and is the gating change before any
   real search is feasible. Keep the cloning path under a feature flag for
   perft cross-checks.
2. **Zobrist hashing + transposition table.** Once apply/undo is in place,
   hash each position incrementally on `apply`. Mandatory for alpha-beta and
   MCTS to amortize repeated sub-trees.
3. **Alpha-beta minimax with a tiny eval** (material first — capitaine = ∞,
   equipier = 1; later add abri usage, capitaine mobility, piece centrality).
   First milestone: beat a uniform random mover at fixed depth.
4. **Self-play harness.** Drive two strategies against each other, log
   move-by-move, surface the final outcomes. Lets us measure changes in eval
   or search without a human in the loop.
5. **Symmetry exploitation.** The board has left/right reflection symmetry
   that's preserved by the initial position. At the root and in opening
   exploration, dedupe mirror-equivalent positions to halve work.

Out-of-band ideas worth keeping on the radar: a bitboard rewrite of the grid
(81 squares fit in `u128`), an MCTS player as a sanity check on the alpha-beta
one, and an opening-book miner that runs deeper perft-style enumeration with
eval cutoffs.
