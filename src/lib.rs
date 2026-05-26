//! ADIX — engine for the abstract strategy game by Echamier Games.
//!
//! Modules:
//! - [`geom`]: board coordinates, directions
//! - [`piece`]: faces, cube orientation, pieces
//! - [`moves`]: move type and slide-direction helper
//! - [`board`]: game state, legal moves, apply, terminal detection
//! - [`notation`]: parse/format moves, render the board

pub mod board;
pub mod geom;
pub mod moves;
pub mod notation;
pub mod perft;
pub mod piece;
pub mod zobrist;
