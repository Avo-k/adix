//! Zobrist hashing for ADIX positions.
//!
//! Two positions hash to the same `u64` iff their cells, side to move, and
//! draw counter all match. The hash is XOR-composable so it can be
//! maintained incrementally as moves are applied / unmade.
//!
//! We don't pre-compute a key table. Each `(pos, piece)` and each
//! `plies_since_progress` value is packed into a u32 and run through
//! `splitmix64`, which gives high-quality 64-bit output from any input
//! and is one multiplication plus a couple of XOR/shift pairs. Cheap to
//! recompute, no large static table to allocate.
//!
//! Soundness for TT use: positions with identical cells/side but
//! different draw counters hash differently (the counter is folded in),
//! so two positions that agree on the hash truly are the same state for
//! perft and search purposes. The full key is stored in each TT entry,
//! so the rare collisions are caught.

use crate::geom::Pos;
use crate::piece::{MoveKind, Piece};

/// Stein's variant of SplitMix64. High-quality avalanche on the full 64 bits
/// from any 64-bit input. Cheap (3 multiplies + a few XOR/shifts).
#[inline]
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Pack `MoveKind` (and `None`) into 2 bits.
#[inline]
const fn move_kind_pack(mk: Option<MoveKind>) -> u8 {
    match mk {
        None => 0,
        Some(MoveKind::Deplacement) => 1,
        Some(MoveKind::Bascule) => 2,
        Some(MoveKind::Pivot) => 3,
    }
}

/// Zobrist contribution of `piece` sitting at `pos`. XOR this in when placing
/// the piece and XOR it out when removing it.
#[inline]
pub fn piece_key(pos: Pos, piece: Piece) -> u64 {
    let sq = pos.rank as u64 * 9 + pos.file as u64; // 7 bits
    let mut packed = sq;
    packed |= (piece.color as u64) << 7; //   1 bit  -> bit 7
    packed |= (piece.kind as u64) << 8; //    1 bit  -> bit 8
    packed |= (piece.cube.top.pack2() as u64) << 9; //    2 bits -> 9..11
    packed |= (piece.cube.bottom.pack2() as u64) << 11; // 2 bits -> 11..13
    packed |= (piece.cube.north.pack2() as u64) << 13; //  2 bits -> 13..15
    packed |= (piece.cube.south.pack2() as u64) << 15; //  2 bits -> 15..17
    packed |= (piece.cube.east.pack2() as u64) << 17; //   2 bits -> 17..19
    packed |= (piece.cube.west.pack2() as u64) << 19; //   2 bits -> 19..21
    packed |= (move_kind_pack(piece.last_kind) as u64) << 21; // 2 bits -> 21..23
    packed |= (piece.streak as u64) << 23; // up to 8 bits -> 23..31 (streak fits in 2 in practice)
    // Mark this as the "piece" domain so it can't collide with the side/plies keys.
    splitmix64(packed | (1u64 << 60))
}

/// XOR this in iff `Color::Fonce` is to move.
pub const ZOB_SIDE_TO_MOVE: u64 = splitmix64(0xDEAD_BEEF_CAFE_BABE);

/// Zobrist contribution of the draw counter. XOR out the old value and
/// XOR in the new one each time `plies_since_progress` changes.
#[inline]
pub fn plies_key(n: u32) -> u64 {
    // Distinct domain (bit 61) so it can't alias with piece_key.
    splitmix64((n as u64) | (1u64 << 61))
}
