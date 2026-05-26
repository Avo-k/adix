//! AlphaZero-style stack for ADIX.
//!
//! - [`encoding`]: pure-Rust state/action encoding, no external deps.
//!   Always compiled, so its tests run with plain `cargo test`.
//! - `net`, `mcts` (gated behind the `tch` cargo feature): the libtorch-
//!   backed policy/value network and PUCT MCTS. Enable with
//!   `cargo build --features tch` — the first build auto-downloads
//!   libtorch (~2 GB) via the `tch/download-libtorch` sub-feature.

pub mod encoding;

pub mod dirichlet;

#[cfg(feature = "tch")]
pub mod net;

#[cfg(feature = "tch")]
pub mod mcts;

#[cfg(feature = "tch")]
pub mod selfplay;

#[cfg(feature = "tch")]
pub mod replay;

#[cfg(feature = "tch")]
pub mod train;
