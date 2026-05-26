//! AlphaZero-style state and action encoding for ADIX.
//!
//! Pure Rust — no `tch` dependency, so this module is always compiled
//! and its tests run under default `cargo test`. The libtorch glue
//! (turning [`encode_state`]'s `&[f32]` into a tensor, decoding the
//! policy head back to a `Move`) lives in [`super::net`] behind the
//! `tch` feature.
//!
//! ## Input planes — 37 × 9 × 9 (`f32`)
//!
//! Always observed from the side-to-move's perspective: own pieces go
//! to "self" planes, opponent's to "opp" planes. Cells in row-major
//! order within each plane (`rank * 9 + file`), planes stacked first.
//!
//! | range  | meaning                                            |
//! |--------|----------------------------------------------------|
//! | 0..1   | self  capitaine / equipier presence                |
//! | 2..3   | opp   capitaine / equipier presence                |
//! | 4..7   | top   face one-hot (Pierre, Feuille, Ciseaux, Abri)|
//! | 8..11  | bottom face one-hot                                |
//! | 12..15 | north  face one-hot                                |
//! | 16..19 | south  face one-hot                                |
//! | 20..23 | east   face one-hot                                |
//! | 24..27 | west   face one-hot                                |
//! | 28..30 | streak one-hot (0, 1, 2)                           |
//! | 31..34 | last_kind one-hot (None, Dep, Bas, Piv)            |
//! | 35     | side-to-move-is-Clair flag (broadcast 0.0 / 1.0)   |
//! | 36     | plies_since_progress / 30 (broadcast)              |
//!
//! ## Action planes — 70 × 9 × 9 = 5670 (`f32`)
//!
//! Indexed `plane * 81 + rank * 9 + file`, where the file/rank is the
//! piece's `from` square.
//!
//! | range  | meaning                                              |
//! |--------|------------------------------------------------------|
//! | 0..64  | deplacement: plane = Dir8.idx() * 8 + (dist - 1)     |
//! | 64..68 | bascule: plane = 64 + Dir4.idx()                     |
//! | 68..70 | pivot:   plane = 68 + RotDir.idx()                   |
//!
//! `Dir8` order: N, S, E, W, NE, NW, SE, SW.
//! `Dir4` order: N, S, E, W.
//! `RotDir` order: Left, Right.
//!
//! Distances run 1..=8 (max slide on a 9×9 board). Indices whose
//! implied `to` square lies off-board are still well-defined slots;
//! they simply won't appear in any legal mask.

use crate::board::{Board, DRAW_PLY_LIMIT};
use crate::geom::{Dir4, Dir8, Pos, RotDir};
use crate::moves::{Move, slide_dir};
use crate::piece::{Arme, Color, Face, Kind, MoveKind};

// --- board geometry constants ---------------------------------------------

pub const BOARD_W: usize = 9;
pub const BOARD_H: usize = 9;
pub const BOARD_CELLS: usize = BOARD_W * BOARD_H; // 81

// --- input planes ---------------------------------------------------------

pub const INPUT_PLANES: usize = 37;
pub const INPUT_SIZE: usize = INPUT_PLANES * BOARD_CELLS; // 2997

const P_SELF_CAP: usize = 0;
const P_SELF_EQ: usize = 1;
const P_OPP_CAP: usize = 2;
const P_OPP_EQ: usize = 3;
const P_TOP: usize = 4; // +4 planes
const P_BOTTOM: usize = 8;
const P_NORTH: usize = 12;
const P_SOUTH: usize = 16;
const P_EAST: usize = 20;
const P_WEST: usize = 24;
const P_STREAK: usize = 28; // +3 planes
const P_LAST_KIND: usize = 31; // +4 planes
const P_STM_CLAIR: usize = 35;
const P_PLIES: usize = 36;

// --- action planes --------------------------------------------------------

pub const ACTION_PLANES: usize = 70;
pub const ACTIONS: usize = ACTION_PLANES * BOARD_CELLS; // 5670

const A_DEP_BASE: usize = 0; // 64 planes
const A_BAS_BASE: usize = 64; // 4 planes
const A_PIV_BASE: usize = 68; // 2 planes

pub const MAX_SLIDE: usize = 8;

// --- direction / face index helpers (private) -----------------------------

#[inline]
fn cell_idx(file: u8, rank: u8) -> usize {
    rank as usize * BOARD_W + file as usize
}

#[inline]
fn dir8_idx(d: Dir8) -> usize {
    match d {
        Dir8::N => 0,
        Dir8::S => 1,
        Dir8::E => 2,
        Dir8::W => 3,
        Dir8::NE => 4,
        Dir8::NW => 5,
        Dir8::SE => 6,
        Dir8::SW => 7,
    }
}

#[inline]
fn dir8_from_idx(i: usize) -> Option<Dir8> {
    Some(match i {
        0 => Dir8::N,
        1 => Dir8::S,
        2 => Dir8::E,
        3 => Dir8::W,
        4 => Dir8::NE,
        5 => Dir8::NW,
        6 => Dir8::SE,
        7 => Dir8::SW,
        _ => return None,
    })
}

#[inline]
fn dir4_idx(d: Dir4) -> usize {
    match d {
        Dir4::N => 0,
        Dir4::S => 1,
        Dir4::E => 2,
        Dir4::W => 3,
    }
}

#[inline]
fn dir4_from_idx(i: usize) -> Option<Dir4> {
    Some(match i {
        0 => Dir4::N,
        1 => Dir4::S,
        2 => Dir4::E,
        3 => Dir4::W,
        _ => return None,
    })
}

#[inline]
fn rot_idx(r: RotDir) -> usize {
    match r {
        RotDir::Left => 0,
        RotDir::Right => 1,
    }
}

#[inline]
fn rot_from_idx(i: usize) -> Option<RotDir> {
    Some(match i {
        0 => RotDir::Left,
        1 => RotDir::Right,
        _ => return None,
    })
}

#[inline]
fn face_idx(f: Face) -> usize {
    match f {
        Face::Arme(Arme::Pierre) => 0,
        Face::Arme(Arme::Feuille) => 1,
        Face::Arme(Arme::Ciseaux) => 2,
        Face::Abri => 3,
    }
}

#[inline]
fn last_kind_idx(lk: Option<MoveKind>) -> usize {
    match lk {
        None => 0,
        Some(MoveKind::Deplacement) => 1,
        Some(MoveKind::Bascule) => 2,
        Some(MoveKind::Pivot) => 3,
    }
}

// --- state encoding -------------------------------------------------------

/// Fill `out` (length [`INPUT_SIZE`]) with the input tensor encoding of
/// `board`, viewed from `board.side_to_move`'s perspective.
pub fn encode_state(board: &Board, out: &mut [f32]) {
    assert_eq!(out.len(), INPUT_SIZE, "out buffer must be {INPUT_SIZE} f32s");
    out.fill(0.0);

    let stm = board.side_to_move;

    for (pos, piece) in board.iter_pieces() {
        let c = cell_idx(pos.file, pos.rank);
        let plane_pres = match (piece.color == stm, piece.kind) {
            (true, Kind::Capitaine) => P_SELF_CAP,
            (true, Kind::Equipier) => P_SELF_EQ,
            (false, Kind::Capitaine) => P_OPP_CAP,
            (false, Kind::Equipier) => P_OPP_EQ,
        };
        out[plane_pres * BOARD_CELLS + c] = 1.0;

        let cube = piece.cube;
        out[(P_TOP + face_idx(cube.top)) * BOARD_CELLS + c] = 1.0;
        out[(P_BOTTOM + face_idx(cube.bottom)) * BOARD_CELLS + c] = 1.0;
        out[(P_NORTH + face_idx(cube.north)) * BOARD_CELLS + c] = 1.0;
        out[(P_SOUTH + face_idx(cube.south)) * BOARD_CELLS + c] = 1.0;
        out[(P_EAST + face_idx(cube.east)) * BOARD_CELLS + c] = 1.0;
        out[(P_WEST + face_idx(cube.west)) * BOARD_CELLS + c] = 1.0;

        // streak: values 0/1/2 are the only reachable ones (3 self-removes).
        let s = piece.streak.min(2) as usize;
        out[(P_STREAK + s) * BOARD_CELLS + c] = 1.0;

        out[(P_LAST_KIND + last_kind_idx(piece.last_kind)) * BOARD_CELLS + c] = 1.0;
    }

    if stm == Color::Clair {
        let base = P_STM_CLAIR * BOARD_CELLS;
        out[base..base + BOARD_CELLS].fill(1.0);
    }

    let p = (board.plies_since_progress as f32) / (DRAW_PLY_LIMIT as f32);
    let base = P_PLIES * BOARD_CELLS;
    out[base..base + BOARD_CELLS].fill(p);
}

/// Allocate-and-encode convenience for one-off callers (e.g. tests).
pub fn encode_state_vec(board: &Board) -> Vec<f32> {
    let mut v = vec![0.0; INPUT_SIZE];
    encode_state(board, &mut v);
    v
}

// --- action encoding ------------------------------------------------------

/// Map a [`Move`] to its flat policy index in `0..ACTIONS`.
///
/// Panics for malformed deplacement moves (non-line `(from, to)` or
/// distance `0` / `> 8`) — those are not produced by the engine's
/// `legal_moves`.
pub fn move_to_index(mv: Move) -> usize {
    match mv {
        Move::Deplacement { from, to } => {
            let (dir, dist) = slide_dir(from, to)
                .expect("deplacement must be a single line");
            assert!(
                (1..=MAX_SLIDE as u8).contains(&dist),
                "deplacement distance out of range: {dist}"
            );
            let plane = A_DEP_BASE + dir8_idx(dir) * MAX_SLIDE + (dist as usize - 1);
            plane * BOARD_CELLS + cell_idx(from.file, from.rank)
        }
        Move::Bascule { from, dir } => {
            let plane = A_BAS_BASE + dir4_idx(dir);
            plane * BOARD_CELLS + cell_idx(from.file, from.rank)
        }
        Move::Pivot { from, rot } => {
            let plane = A_PIV_BASE + rot_idx(rot);
            plane * BOARD_CELLS + cell_idx(from.file, from.rank)
        }
    }
}

/// Inverse of [`move_to_index`]. Returns `None` when the index would
/// produce a deplacement whose `to` square lies off the 9×9 board.
/// Otherwise returns the corresponding `Move` shape — note this does
/// **not** verify legality (use [`Board::legal_moves`] for that).
pub fn index_to_move(idx: usize) -> Option<Move> {
    if idx >= ACTIONS {
        return None;
    }
    let plane = idx / BOARD_CELLS;
    let cell = idx % BOARD_CELLS;
    let file = (cell % BOARD_W) as u8;
    let rank = (cell / BOARD_W) as u8;
    let from = Pos::new(file, rank);

    if plane < A_BAS_BASE {
        // deplacement
        let p = plane - A_DEP_BASE;
        let dir = dir8_from_idx(p / MAX_SLIDE)?;
        let dist = (p % MAX_SLIDE) as i8 + 1;
        let (df, dr) = dir.delta();
        let to = from.offset(df * dist, dr * dist)?;
        Some(Move::Deplacement { from, to })
    } else if plane < A_PIV_BASE {
        let dir = dir4_from_idx(plane - A_BAS_BASE)?;
        Some(Move::Bascule { from, dir })
    } else {
        let rot = rot_from_idx(plane - A_PIV_BASE)?;
        Some(Move::Pivot { from, rot })
    }
}

/// Fill `mask` (length [`ACTIONS`]) with 1.0 at legal-move indices,
/// 0.0 elsewhere. Cheap: O(legal_moves).
pub fn fill_legal_mask(board: &Board, mask: &mut [f32]) {
    assert_eq!(mask.len(), ACTIONS, "mask buffer must be {ACTIONS} f32s");
    mask.fill(0.0);
    let mut moves = Vec::with_capacity(64);
    board.legal_moves_into(&mut moves);
    for mv in moves {
        mask[move_to_index(mv)] = 1.0;
    }
}

// --- tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Dir4, Pos, RotDir};

    #[test]
    fn sizes_match_advertised() {
        assert_eq!(INPUT_PLANES * BOARD_CELLS, INPUT_SIZE);
        assert_eq!(ACTION_PLANES * BOARD_CELLS, ACTIONS);
        assert_eq!(ACTIONS, 5670);
    }

    #[test]
    fn encode_state_initial_position_basics() {
        let board = Board::initial();
        let v = encode_state_vec(&board);
        // Plane sums should be sane.
        let plane_sum = |p: usize| -> f32 {
            v[p * BOARD_CELLS..(p + 1) * BOARD_CELLS].iter().sum()
        };
        // 1 self capitaine, 9 equipiers per side at the start.
        assert_eq!(plane_sum(P_SELF_CAP), 1.0);
        assert_eq!(plane_sum(P_SELF_EQ), 9.0);
        assert_eq!(plane_sum(P_OPP_CAP), 1.0);
        assert_eq!(plane_sum(P_OPP_EQ), 9.0);
        // 20 pieces total → each of the 6 face-plane groups must sum to 20.
        for base in [P_TOP, P_BOTTOM, P_NORTH, P_SOUTH, P_EAST, P_WEST] {
            let group: f32 = (base..base + 4).map(plane_sum).sum();
            assert_eq!(group, 20.0, "face group at {base} should cover all 20 pieces");
        }
        // streak: every piece starts at 0.
        assert_eq!(plane_sum(P_STREAK), 20.0);
        assert_eq!(plane_sum(P_STREAK + 1), 0.0);
        assert_eq!(plane_sum(P_STREAK + 2), 0.0);
        // last_kind: every piece starts with None.
        assert_eq!(plane_sum(P_LAST_KIND), 20.0);
        // side-to-move is Clair at the start → broadcast plane = all 1.0.
        assert_eq!(plane_sum(P_STM_CLAIR), 81.0);
        // plies_since_progress = 0 at start.
        assert_eq!(plane_sum(P_PLIES), 0.0);
    }

    #[test]
    fn move_index_roundtrip_covers_action_space() {
        // For each plane × cell, building a Move from the index and
        // converting back must give the same index (when the implied
        // `to` square is on-board).
        for idx in 0..ACTIONS {
            if let Some(mv) = index_to_move(idx) {
                assert_eq!(
                    move_to_index(mv),
                    idx,
                    "round-trip failed at index {idx}: {mv:?}"
                );
            }
        }
    }

    #[test]
    fn move_index_explicit_layout() {
        // a1 = (0, 0), plane 0 = deplacement Dir8::N, dist 1 → index 0.
        let from = Pos::new(0, 0);
        let to = Pos::new(0, 1);
        assert_eq!(
            move_to_index(Move::Deplacement { from, to }),
            0
        );
        // Same square, Dir8::N dist 2 → plane 1, index 1*81 + 0 = 81.
        let to2 = Pos::new(0, 2);
        assert_eq!(
            move_to_index(Move::Deplacement { from, to: to2 }),
            81
        );
        // Bascule N from a1: plane 64, cell 0 → idx 64*81 = 5184.
        assert_eq!(
            move_to_index(Move::Bascule { from, dir: Dir4::N }),
            64 * 81
        );
        // Pivot Right from a1: plane 69, idx 69*81 = 5589.
        assert_eq!(
            move_to_index(Move::Pivot { from, rot: RotDir::Right }),
            69 * 81
        );
    }

    #[test]
    fn legal_mask_at_initial_position_matches_legal_moves() {
        let board = Board::initial();
        let mut mask = vec![0.0_f32; ACTIONS];
        fill_legal_mask(&board, &mut mask);
        let set_bits: f32 = mask.iter().sum();
        let n_legal = board.legal_moves().len() as f32;
        assert_eq!(set_bits, n_legal);
        // Every legal move's index must be set; nothing else.
        for mv in board.legal_moves() {
            assert_eq!(mask[move_to_index(mv)], 1.0);
        }
    }

    #[test]
    fn index_to_move_rejects_off_board_targets() {
        // a1 + Dir8::W = off-board: plane 3 (W) × 8 dist + 0 (dist 1)
        // = plane 24, cell 0 → idx 24*81 = 1944.
        let idx = 24 * 81;
        assert!(index_to_move(idx).is_none());
    }

    #[test]
    fn stm_perspective_swaps_self_and_opp_planes() {
        let mut board = Board::initial();
        // Apply any legal move to flip side-to-move.
        let mv = board.legal_moves()[0];
        board.apply_legal(mv);
        let v = encode_state_vec(&board);
        let plane_sum = |p: usize| -> f32 {
            v[p * BOARD_CELLS..(p + 1) * BOARD_CELLS].iter().sum()
        };
        // Now Fonce is to move — Fonce's pieces are "self", Clair's are "opp".
        // Counts are still roughly 1 cap + 9 eq each side.
        assert_eq!(plane_sum(P_SELF_CAP), 1.0);
        assert_eq!(plane_sum(P_OPP_CAP), 1.0);
        // STM plane is now 0 (Fonce, not Clair).
        assert_eq!(plane_sum(P_STM_CLAIR), 0.0);
    }
}
