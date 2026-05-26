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
cargo build              # debug build of lib + bins
cargo test               # all tests (unit + integration)
cargo test <name>        # single test by substring, e.g. cargo test capture_blocked
cargo test --test rules  # one integration file (tests/rules.rs)
cargo run                # launch the REPL on the initial position
cargo run --release --bin perft     -- 5 [--search|--tt[=mb]]
cargo run --release --bin selfplay  -- <white> <black> [N] [--swap]
cargo build --release    # optimized build, useful before tree-search work
```

The REPL accepts: `help`, `board`, `moves`, `moves <sq>`, `<move>`, `undo`,
`quit`. Move notation: `e1-e2` (deplacement), `e1>n` (bascule), `e1@l|r` (pivot).
Board glyphs: `O` pierre, `+` feuille, `X` ciseaux, `^` abri; `w`/`b` color
prefix; `*` marks a capitaine.

Agent specs for `selfplay`:
- `random`
- `ab:<depth>` — alpha-beta, full positional eval
- `ab-mat:<depth>` — alpha-beta, material-only (for A/B comparison)
- `mcts:<iterations>` — MCTS, fixed iteration budget
- `mcts-t:<ms>` — MCTS, fixed time budget in milliseconds

`--swap` alternates colors between games so the result isn't dominated
by who moves first. Per-move timing reference: `mcts:5000` ≈ 65 ms/move
(range 18-104 ms depending on position), `ab:3` ≈ 80 ms/move with the
full eval.

## Vocabulary convention (load-bearing)

**Code uses French terms without accents** for game-domain types: `capitaine`,
`equipier`, `pierre`, `feuille`, `ciseaux`, `abri`, `bascule`, `pivot`,
`echamier`, `deplacement`, `clair`/`fonce` (white/black). Do not anglicize
them. Adding accents (e.g. `équipier`) will break identifiers.

## Architecture

Seven modules, kept deliberately small and dependency-free (no external crates):

- `geom` — `Pos { file, rank }` with file 0–8 = a–i, rank 0–8 = 1–9. `Pos::is_dark`
  uses `(file + rank) % 2 == 0`, so a1 (the dark corner) is dark.
- `piece` — `Arme`, `Face` (`Arme | Abri`), `Cube` (6 oriented faces),
  `Piece` carrying `last_kind: Option<MoveKind>` + `streak: u8` for the 3× rule.
- `moves` — `Move` enum + `slide_dir` helper that classifies a `(from, to)` pair.
- `board` — `Board` holds the 9×9 grid, side to move, `plies_since_progress`
  draw counter, captured list, alive counts, own-piece bitboards, and Zobrist
  hash. `legal_moves`, `apply`, `apply_legal`+`unmake`, `outcome` live here.
- `notation` — parsing + ASCII rendering.
- `zobrist` — splitmix64-derived keys for incremental Zobrist hashing.
- `perft` — leaf-counter + `PerftTT` transposition table.
- `eval` — positional evaluation terms (material, threats on capitaine,
  capitaine confinement, RPS arme imbalance, mobility, offensive
  threats) + `full_eval` aggregator.
- `agent` — `Player` trait + `RandomPlayer` + `AlphaBetaPlayer` (uses
  `eval::full_eval` by default, `material_eval` available via
  `new_material_only` for A/B testing) + `MctsPlayer` (UCT) +
  `play_game` harness.

The library is in `src/lib.rs`; the REPL is `src/bin/adix.rs`, the perft
benchmark is `src/bin/perft.rs`, and the self-play harness is
`src/bin/selfplay.rs` — three separate binary targets.

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

From `Board::initial()`, release build. Three modes, all validate the same
locked node counts:

- **Bulk-count** (`perft`): at depth 1, return `legal_moves().len()` directly.
  Used in [tests/perft.rs](tests/perft.rs); also the default benchmark.
- **Search mode** (`perft_search`, `perft <d> --search`): always applies and
  unmakes every move down to depth 0. Representative of what alpha-beta will
  actually do per node.
- **TT mode** (`perft_tt`, `perft <d> --tt[=mb]`): bulk-count + transposition
  table keyed on Zobrist hash. Default 64 MB; pass e.g. `--tt=256` to widen.

| depth | nodes        | bulk-count | search-mode | TT (64 MB) | bulk Mn/s | search Mn/s | TT Mn/s |
|------:|-------------:|-----------:|------------:|-----------:|----------:|------------:|--------:|
| 3     | 82 110       | <1 ms      | 2 ms        | <1 ms      | ~430      | ~41         | ~445    |
| 4     | 3 811 526    | 8 ms       | 95 ms       | 8 ms       | ~450      | ~40         | ~450    |
| 5     | 194 027 791  | 0.42 s     | 4.8 s       | 0.26 s     | ~465      | ~40         | ~735    |
| 6     | 9 830 027 851| 22 s       | —           | 8.6 s (256 MB) | ~450  | —           | ~1100   |

The Zobrist maintenance costs roughly 10 % in bulk-count and ~50 % in
search-mode against the pre-Zobrist engine — that's the upper bound of
the per-node tax. TT hit rates climb fast with depth: ~35 % at depth 5,
~38 % at depth 6, which is why TT mode crosses bulk-count around depth 5.

Branching factor settles around ~51. Depths 0–3 are locked in
[tests/perft.rs](tests/perft.rs) (all three modes share the same expected
numbers); depths 4–6 are run on demand via
`./target/release/perft <max_depth> [divide_at] [--search|--tt[=mb]]`.
**If you touch move generation, re-run perft 4 — a single off-by-one
shows up loudly here.**

### Hot-path architecture

The engine has two apply paths:

- `Board::apply(mv) -> Result<Option<Outcome>, IllegalMove>` — the public,
  validating one. Used by the REPL. Internally delegates to `apply_legal`
  after `validate`.
- `Board::apply_legal(mv) -> Undo` and `Board::unmake(mv, Undo)` — the
  hot path. Skips validation (caller guarantees legality, e.g. via
  `legal_moves`), returns an `Undo` token, and `unmake` reverses it
  exactly. This is what perft (and any future search) uses.

Three pieces of state are maintained incrementally to keep per-node work O(1):

- `alive: [[u8; 2]; 2]` — `[color][kind]` live counts. `outcome()` reads
  these directly instead of scanning 81 cells.
- `own_bb: [u128; 2]` — per-color occupancy bitboard, bit `rank*9 + file`.
  `legal_moves_into` iterates set bits (~10 ops) instead of scanning cells.
- `zobrist: u64` — 64-bit hash of (cells, side_to_move,
  plies_since_progress). Keys are derived on the fly via `splitmix64` on a
  packed `(pos, piece-state)` u64 — no pre-allocated key table. Side and
  plies have their own domain bits to avoid aliasing.

The mutation choke points are `set` and `take`; both maintain `own_bb`
and the Zobrist contribution. The one place we bypass them is the
pivot-in-place path in `apply_legal` (piece stays on the same square,
same color — bitboard bit unchanged; Zobrist is XORed manually). The
side-to-move and plies-counter contributions to Zobrist are XORed at the
end of `apply_legal` / `unmake`.

The Zobrist round-trip is verified by [tests/zobrist.rs](tests/zobrist.rs):
`apply_legal` then `unmake` must restore the hash exactly, and the
incremental hash must always match `zobrist_from_scratch()`. **If you
add a new mutation path, route it through `set`/`take` or it will silently
desync the hash and the TT will return wrong perft counts.**

The transposition table for perft is in [src/perft.rs](src/perft.rs):
`PerftTT` is a fixed-size always-replace table keyed on
`(zobrist & mask)`, with the full Zobrist stored in each entry so
collisions are caught. The bulk-count short-circuit at depth 1 is
unchanged — TT probes happen for depth ≥ 2.

## Agents

[src/agent.rs](src/agent.rs) defines:

- `Player` trait — `choose_move(&Board) -> Option<Move>` + `name()`.
- `RandomPlayer` — uniform random over `legal_moves()`. Seedable via
  splitmix64.
- `AlphaBetaPlayer` — fixed-depth negamax with α/β cutoffs. Leaf eval
  defaults to [`eval::full_eval`](src/eval.rs) (material + positional
  terms). Terminal nodes score `MATE - ply` so the engine prefers
  faster wins. Search heuristics:
  - **MVV-LVA move ordering** (captures first, sorted by victim value).
  - **Killer-move table**, 2 slots/ply for quiet moves that recently
    caused β-cutoffs at the same depth.
  - **Persistent transposition table** (2^17 ≈ 4 MB) keyed on Zobrist
    with `Exact / Lower / Upper` bound flags. The TT seeds the first
    move tried at each node when its MVV-LVA score beats the natural
    top of the list (a quiet TT-best ahead of a real capture loses
    cutoffs, so we guard the swap).
  - **Quiescence search** at `depth==0`: instead of returning the
    static eval, run a capture-only search until the position is
    quiet, then eval. Uses standard stand-pat α/β: at every qnode,
    static eval is a lower bound (we always have the option not to
    capture), so `eval ≥ β` cuts immediately. Captures are MVV-ordered.
    Resolves the "horizon noise" that previously kept `ab:5 vs ab:3`
    at ~50% — without quiescence the eval sees a half-completed
    exchange and mis-rates the leaf.

  Ties at the root are broken randomly (different seed per game) so
  two AB players at the same depth don't loop. `new_material_only(...)`
  swaps in the material-only leaf eval and tags itself `ab-mat` for
  direct A/B comparison. Per-move stats exposed via `last_nodes` (main
  search), `last_qnodes` (quiescence), `last_elapsed_ms`,
  `last_tt_probes`, `last_tt_hits`.
- `MctsPlayer` — textbook UCT (`c = √2`) with random rollouts. Arena-
  based tree (one `Vec<MctsNode>`). Picks the most-visited root child
  (more robust than best-mean for small budgets). Memory scales with
  iteration count. Budget is an `MctsBudget` enum: `Iterations(n)` (fixed
  iteration count) or `TimeMs(ms)` (run until at least `ms` have elapsed,
  clock checked every 64 iters). `last_iterations` / `last_elapsed_ms`
  report what the last move actually consumed.
- `play_game(clair, fonce)` — drives a game from the initial position.
  Uses the validating `apply` (the harness isn't perf-critical) and
  panics if a player returns an illegal move.

### Baselines (W-L over 20 games with `--swap`, recent run)

With the positional eval (default `ab`):

| matchup | result | s/game |
|---|---|---|
| **ab:1 vs ab-mat:3** | **20-0** | <0.01 |
| **ab:3 vs ab-mat:3** | **20-0** | ~0.9 |
| **ab:2 vs mcts:5000** | **20-0** | ~1.5 |
| **ab:3 vs mcts:5000** | **20-0** | ~2.0 |
| **ab:3 vs mcts-t:100** (iso-time) | **19-1** | ~2.9 |
| **ab:3 vs mcts-t:1000** (MCTS 10× thinking) | **10-0** (10 games) | ~15.2 |
| ab:3 vs ab:1 | 15-5 | ~2.2 |

For reference, the *material-only* era looked like this:

| matchup | result | s/game |
|---|---|---|
| ab-mat:5 vs random | 20-0 | ~5.0 |
| ab-mat:4 vs ab-mat:2 | 9-11 | ~0.04 (depth didn't help) |
| mcts:5000 vs ab-mat:3 | 14-6 (AB lost) | ~1.9 |
| mcts:10000 vs ab-mat:5 | 9-1 (AB lost, 10 games) | ~5.1 |

The positional eval flipped both the MCTS comparison *and* shallow-vs-deep
scaling: `ab:1` (full eval) beats `ab-mat:3` (deeper but blind),
confirming the bottleneck was eval quality, not search depth. The
biggest contributors are probably `threats_on_capitaine`
(near-MATE signal when undefended) and `mobility_differential`
(prevents the engine sitting on its hands).

Two caveats worth tracking. (1) The eval is heavy: each call does
~12 `moves_for_into` invocations (one per term × two sides), so
search time grows much faster with depth than with the material-only
eval — `ab:4` is now ~10× slower per move than `ab-mat:4`. (2)
Preliminary `ab:4` vs `ab:2` was *not* a clean win for `ab:4` — early
games leaned toward `ab:2`, suggesting the eval may have noisy
positional terms (mobility flickers move-to-move) that amplify with
depth. Worth investigating before tuning weights blindly.

All weights are in [src/eval.rs](src/eval.rs) and are not yet
selfplay-tuned — first-pass guesses.

### AB search-heuristic speedups

Measured on the early-middlegame opening of `ab:N vs random` (release
build, 4 measured moves, full positional eval). Effective branching
factor = `nodes(d) / nodes(d-1)`.

| depth | baseline (α/β only) | + MVV-LVA + killers | + persistent TT | + quiescence |
|------:|--------------------:|--------------------:|----------------:|-------------:|
| 3 | 26 548 / 51 ms | 6 319 / 11 ms | 5 645 / 11 ms | 5 738 main + 5 397 q / 10 ms |
| 4 | 476 845 / 903 ms | 26 974 / 49 ms | 32 037 / 60 ms | 35 517 main + 30 673 q / 65 ms |
| 5 | n/a (too slow) | 406 623 / 800 ms | 343 118 / 703 ms | 288 549 main + 265 686 q / 564 ms |

The big win is **move ordering** — MVV-LVA + killers drops the
effective branching factor at d=4 from ~18 to ~4.3, close to the
α/β-optimal `√50 ≈ 7`. The TT (with bound flags + best-move hint
constrained to "beat the natural MVV-LVA top") helps mostly at d=5+
where transpositions accumulate; at d=4 it's roughly neutral.

**Quiescence** roughly doubles the total node count (qnodes ≈ 47% of
all nodes) but each qnode is cheap (no TT probe/store), so combined
throughput nearly doubles to ~1000 knodes/s. The depth-5 wall time
actually *dropped* from 703 to 564 ms because the qnode work
amortizes over more cheap nodes per ms.

### Playing-strength dynamics post-quiescence

Quiescence is supposed to fix the horizon-noise we saw before
(`ab:5 vs ab:3` was 5-5 with material-only-ish eval). The story
turned out more interesting:

| matchup | with quiescence | without quiescence |
|---|---|---|
| ab:5 vs ab:1 | 7-3, 277 plies | (not measured) |
| ab:5 vs ab:3 | 4-6, **557 plies** | 5-5, 169 plies |
| ab:5 vs mcts-t:500 (iso-time-ish) | 5-1, 241 plies | 6-0, 34 plies |

`ab:5 vs ab:3` got *longer*, not more decisive: average 557 plies
with 14.7 captures and only 1-2 equipiers left per side at the end.
Quiescence let both sides defend tactical exchanges much better, so
games dragged into deep endgames where the small d=2 depth gap
doesn't matter — both engines see the same exchanges via qsearch.
A bigger depth gap (ab:5 vs ab:1) still produces a measurable win
(7-3) but no longer a rout.

Read: quiescence made *every* depth dramatically stronger by
auto-resolving tactical follow-up. The d=5 vs d=3 differential
shrank because both already play tactically well. To break further
out of this plateau, the next interesting frontier is the **eval**
(positional terms that distinguish quiet positions) rather than
more search.

### Game dynamics (20-game samples, `--swap`)

| matchup | plies | captures | 1st cap | by cap | by eq elim | winner eq / loser eq |
|---|-:|-:|-:|-:|-:|-:|
| random vs random | 74 | 0.8 | ply 39 | 90% | 10% | 5.2 / 5.0 |
| ab:3 vs random | 21 | 2.6 | ply 8 | 100% | 0% | 8.7 / 6.9 |
| ab:3 vs ab:1 | 58 | 10.3 | ply 7 | 75% | 25% | 4.9 / 3.2 |
| ab:3 vs ab:3 | 58 | 8.4 | ply 11 | 100% | 0% | 5.2 / 5.2 |

Two non-obvious patterns:

1. **Stronger ≠ shorter in general.** Strong-vs-much-weaker is very
   short (~21 plies, almost no attrition) — the strong side rushes the
   capitaine. But strong-vs-slightly-weaker and strong-vs-mirror are
   *longer than random play* (~58 vs 74 plies) because both sides are
   competent enough to defend; the games turn into tactical attrition
   with 8-10 captures.
2. **Capitaine is the dominant win condition** between engines
   (75-100% of wins), even when 10+ equipiers have already fallen.
   Equipier-elimination wins (25%) appear only in peer-ish play where
   neither side finds the tactic first.

## Where this is going (big next steps)

In rough priority order:

1. **Quiescence search.** Deeper AB doesn't reliably beat shallower AB
   (e.g., `ab:5 vs ab:3` was 5-5 in a recent 10-game match). The eval
   is *noisy* at the search horizon because positional terms like
   `mobility_differential` and `offensive_threats` swing wildly when a
   capture is one ply outside the search. Standard fix: at depth==0,
   instead of returning eval, run a small capture-only search until
   the position is "quiet" (no pending captures), *then* eval. This is
   the single biggest expected improvement for tactical strength now.
2. **Defended-pieces / SEE-lite eval term.** For each of my pieces
   under threat by an opp piece P, check whether a same-arme defender
   can recapture P. Naive O(N²) — needs care to stay cheap at search
   leaves.
3. **Iterative deepening + time control.** Necessary for any practical
   "play one move within T seconds" interface. Free quality win
   because the TT-best from shallow searches seeds the deeper ones.
4. **Selfplay-tune the eval weights.** Current constants in
   [src/eval.rs](src/eval.rs) are guesses. A coordinate-descent
   tournament against MCTS could refine each weight independently.
5. **Symmetry exploitation.** The board has left/right reflection
   symmetry that's preserved by the initial position. At the root and
   in opening exploration, dedupe mirror-equivalent positions to halve
   work.

Out-of-band ideas worth keeping on the radar: full bitboard move-gen
(slides via shift+mask on `u128`), a shrunk `Piece` representation (Cube
currently ~12 bytes; could be packed to one `u32`), an MCTS player as a
sanity check on the alpha-beta one, and an opening-book miner.
