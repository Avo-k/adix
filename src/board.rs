use crate::geom::{Dir4, Dir8, Pos, RotDir};
use crate::moves::{Move, slide_dir};
use crate::piece::{
    Arme, Color, Kind, MoveKind, Piece, starting_capitaine_cube, starting_equipier_cube,
};
use crate::zobrist::{ZOB_SIDE_TO_MOVE, piece_key, plies_key};

const SIZE: usize = 9;
pub const DRAW_PLY_LIMIT: u32 = 30;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Outcome {
    Win(Color),
    Draw,
}

#[derive(Debug)]
pub enum IllegalMove {
    NoPieceThere,
    NotYourTurn,
    PieceUnderAbriCannotSlide,
    PieceUnderAbriCannotPivot,
    BasculeOntoOccupied,
    BasculeOffBoard,
    NotALine,
    WrongDirectionForArme,
    DistanceTooFarForCapitaine,
    PathBlocked,
    FriendlyAtDestination,
    CannotBeatTarget,
    TargetUnderAbri,
    OffBoard,
}

/// Information needed to reverse `apply_legal`.
#[derive(Clone, Debug)]
pub struct Undo {
    moved_piece_pre: Piece,
    captured: Option<(Pos, Piece)>,
    self_removed: bool,
    plies_since_progress_pre: u32,
}

#[derive(Clone, Debug)]
pub struct Board {
    cells: [[Option<Piece>; SIZE]; SIZE],
    pub side_to_move: Color,
    pub plies_since_progress: u32,
    pub ply: u32,
    pub captured: Vec<Piece>,
    /// Live piece counts indexed `[color as usize][kind as usize]`. Maintained
    /// incrementally so `outcome()` is O(1) instead of scanning the board.
    alive: [[u8; 2]; 2],
    /// Per-color occupancy bitboard: bit `rank*9 + file` is set iff that color
    /// has a piece on that square. Move-gen iterates set bits to enumerate
    /// own pieces (~10 ops) instead of scanning 81 cells.
    own_bb: [u128; 2],
    /// Zobrist hash of (cells, side_to_move, plies_since_progress). Maintained
    /// incrementally by `set` / `take` / the pivot-in-place path / and the
    /// side & plies updates in `apply_legal` and `unmake`.
    pub zobrist: u64,
}

impl Board {
    pub fn empty() -> Self {
        // Clair to move, plies_since_progress = 0, board empty.
        // Side-to-move key isn't XORed in (Clair = "no flip").
        let zobrist = plies_key(0);
        Self {
            cells: Default::default(),
            side_to_move: Color::Clair,
            plies_since_progress: 0,
            ply: 0,
            captured: Vec::new(),
            alive: [[0; 2]; 2],
            own_bb: [0; 2],
            zobrist,
        }
    }

    /// Empty board used by tests to set up specific positions.
    pub fn empty_for_test() -> Self {
        Self::empty()
    }

    pub fn force_place(&mut self, p: Pos, piece: Piece) {
        if self.at(p).is_none() {
            self.alive[piece.color as usize][piece.kind as usize] += 1;
        }
        self.set(p, piece);
    }

    pub fn set_side_to_move(&mut self, c: Color) {
        self.side_to_move = c;
    }

    pub fn initial() -> Self {
        let mut b = Self::empty();
        let cap = starting_capitaine_cube();
        let eq = starting_equipier_cube();

        // White (clair) at south.
        let white_cap = Pos::new(4, 0); // e1
        let white_eqs: [Pos; 9] = [
            Pos::new(2, 0), Pos::new(6, 0),                                    // c1 g1
            Pos::new(1, 1), Pos::new(3, 1), Pos::new(5, 1), Pos::new(7, 1),    // b2 d2 f2 h2
            Pos::new(2, 2), Pos::new(4, 2), Pos::new(6, 2),                    // c3 e3 g3
        ];
        b.force_place(white_cap, Piece::new(Color::Clair, Kind::Capitaine, cap));
        for p in white_eqs {
            b.force_place(p, Piece::new(Color::Clair, Kind::Equipier, eq));
        }

        // Black (foncé) at north.
        let black_cap = Pos::new(4, 8); // e9
        let black_eqs: [Pos; 9] = [
            Pos::new(2, 8), Pos::new(6, 8),                                    // c9 g9
            Pos::new(1, 7), Pos::new(3, 7), Pos::new(5, 7), Pos::new(7, 7),    // b8 d8 f8 h8
            Pos::new(2, 6), Pos::new(4, 6), Pos::new(6, 6),                    // c7 e7 g7
        ];
        b.force_place(black_cap, Piece::new(Color::Fonce, Kind::Capitaine, cap));
        for p in black_eqs {
            b.force_place(p, Piece::new(Color::Fonce, Kind::Equipier, eq));
        }

        b
    }

    /// Recompute the Zobrist hash from scratch. Only used as a sanity check
    /// in tests / debug builds — the engine relies on incremental updates.
    pub fn zobrist_from_scratch(&self) -> u64 {
        let mut z = plies_key(self.plies_since_progress);
        if matches!(self.side_to_move, Color::Fonce) {
            z ^= ZOB_SIDE_TO_MOVE;
        }
        for r in 0..SIZE {
            for f in 0..SIZE {
                if let Some(piece) = &self.cells[r][f] {
                    let pos = Pos::new(f as u8, r as u8);
                    z ^= piece_key(pos, *piece);
                }
            }
        }
        z
    }

    pub fn at(&self, p: Pos) -> Option<&Piece> {
        self.cells[p.rank as usize][p.file as usize].as_ref()
    }
    fn at_mut(&mut self, p: Pos) -> &mut Option<Piece> {
        &mut self.cells[p.rank as usize][p.file as usize]
    }
    #[inline]
    fn bb_bit(p: Pos) -> u128 {
        1u128 << (p.rank as u32 * 9 + p.file as u32)
    }
    /// Place `piece` at `p`. Assumes `p` is empty (caller should `take` first
    /// if not). Maintains the own-piece bitboard and Zobrist hash.
    fn set(&mut self, p: Pos, piece: Piece) {
        debug_assert!(self.at(p).is_none(), "set onto occupied square");
        self.own_bb[piece.color as usize] |= Self::bb_bit(p);
        self.zobrist ^= piece_key(p, piece);
        *self.at_mut(p) = Some(piece);
    }
    /// Remove and return the piece at `p`. Maintains the own-piece bitboard
    /// and Zobrist hash.
    fn take(&mut self, p: Pos) -> Option<Piece> {
        let piece = self.at_mut(p).take()?;
        self.own_bb[piece.color as usize] &= !Self::bb_bit(p);
        self.zobrist ^= piece_key(p, piece);
        Some(piece)
    }

    pub fn iter_pieces(&self) -> impl Iterator<Item = (Pos, &Piece)> {
        (0..SIZE).flat_map(move |r| {
            (0..SIZE).filter_map(move |f| {
                let p = Pos::new(f as u8, r as u8);
                self.at(p).map(|piece| (p, piece))
            })
        })
    }

    /// Counts (capitaines_alive, equipiers_alive) for the given color. O(1).
    pub fn alive_counts(&self, color: Color) -> (u32, u32) {
        let row = &self.alive[color as usize];
        (row[Kind::Capitaine as usize] as u32, row[Kind::Equipier as usize] as u32)
    }

    pub fn outcome(&self) -> Option<Outcome> {
        let w = &self.alive[Color::Clair as usize];
        let b = &self.alive[Color::Fonce as usize];
        if w[Kind::Capitaine as usize] == 0 || w[Kind::Equipier as usize] == 0 {
            return Some(Outcome::Win(Color::Fonce));
        }
        if b[Kind::Capitaine as usize] == 0 || b[Kind::Equipier as usize] == 0 {
            return Some(Outcome::Win(Color::Clair));
        }
        if self.plies_since_progress >= DRAW_PLY_LIMIT {
            return Some(Outcome::Draw);
        }
        None
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::new();
        self.legal_moves_into(&mut out);
        out
    }

    /// Append all legal moves for the side to move to `out`. The buffer is not
    /// cleared — callers reuse it across nodes to avoid per-node allocation.
    pub fn legal_moves_into(&self, out: &mut Vec<Move>) {
        self.moves_for_into(self.side_to_move, out);
    }

    /// Append all *hypothetical* moves for `color` to `out`, regardless of
    /// whose turn it actually is. Used by the evaluator to assess threats
    /// and mobility from a non-side-to-move perspective. The path-clear,
    /// arme-direction, and capture rules are identical to `legal_moves_into`.
    pub fn moves_for_into(&self, color: Color, out: &mut Vec<Move>) {
        let mut bb = self.own_bb[color as usize];
        while bb != 0 {
            let sq = bb.trailing_zeros();
            bb &= bb - 1;
            let f = (sq % 9) as u8;
            let r = (sq / 9) as u8;
            let piece = self.cells[r as usize][f as usize]
                .as_ref()
                .expect("own_bb bit set without piece");
            self.gen_moves_from(Pos::new(f, r), piece, out);
        }
    }

    pub fn legal_moves_from(&self, from: Pos) -> Vec<Move> {
        let mut out = Vec::new();
        if let Some(piece) = self.at(from)
            && piece.color == self.side_to_move
        {
            self.gen_moves_from(from, piece, &mut out);
        }
        out
    }

    fn gen_moves_from(&self, pos: Pos, piece: &Piece, out: &mut Vec<Move>) {
        // Pivots (unless under abri).
        if !piece.is_under_abri() {
            out.push(Move::Pivot { from: pos, rot: RotDir::Left });
            out.push(Move::Pivot { from: pos, rot: RotDir::Right });
        }

        // Bascules: must land on an empty in-board square.
        for d in Dir4::ALL {
            let (df, dr) = d.delta();
            if let Some(target) = pos.offset(df, dr)
                && self.at(target).is_none()
            {
                out.push(Move::Bascule { from: pos, dir: d });
            }
        }

        // Deplacements: only when an arme is active.
        let Some(arme) = piece.active_arme() else { return };
        let dirs: &[Dir8] = match arme {
            Arme::Pierre => &[
                Dir8::N, Dir8::S, Dir8::E, Dir8::W,
                Dir8::NE, Dir8::NW, Dir8::SE, Dir8::SW,
            ],
            Arme::Feuille => &Dir8::ORTHO,
            Arme::Ciseaux => &Dir8::DIAG,
        };
        let max_steps = match piece.kind {
            Kind::Capitaine => 1,
            Kind::Equipier => 8,
        };
        for &dir in dirs {
            let (df, dr) = dir.delta();
            let mut cur = pos;
            for _ in 0..max_steps {
                let Some(next) = cur.offset(df, dr) else { break };
                cur = next;
                match self.at(cur) {
                    None => out.push(Move::Deplacement { from: pos, to: cur }),
                    Some(other) => {
                        if other.color == piece.color {
                            // friendly: blocked, no landing
                        } else {
                            // enemy: capture iff not under abri and arme beats theirs
                            if !other.is_under_abri()
                                && let Some(b) = other.active_arme()
                                && arme.beats(b)
                            {
                                out.push(Move::Deplacement { from: pos, to: cur });
                            }
                        }
                        break; // always blocks further sliding
                    }
                }
            }
        }
    }

    pub fn apply(&mut self, mv: Move) -> Result<Option<Outcome>, IllegalMove> {
        self.validate(mv)?;
        self.apply_legal(mv);
        Ok(self.outcome())
    }

    /// Validate `mv` against rules without touching the board. Returns the
    /// same errors `apply` used to raise. Called by `apply`; `apply_legal`
    /// (the hot path) skips this — it trusts the caller.
    fn validate(&self, mv: Move) -> Result<(), IllegalMove> {
        let from = mv.from();
        let piece = self.at(from).copied().ok_or(IllegalMove::NoPieceThere)?;
        if piece.color != self.side_to_move {
            return Err(IllegalMove::NotYourTurn);
        }
        match mv {
            Move::Pivot { .. } => {
                if piece.is_under_abri() {
                    return Err(IllegalMove::PieceUnderAbriCannotPivot);
                }
            }
            Move::Bascule { dir, .. } => {
                let (df, dr) = dir.delta();
                let target = from.offset(df, dr).ok_or(IllegalMove::BasculeOffBoard)?;
                if self.at(target).is_some() {
                    return Err(IllegalMove::BasculeOntoOccupied);
                }
            }
            Move::Deplacement { from: _, to } => {
                if piece.is_under_abri() {
                    return Err(IllegalMove::PieceUnderAbriCannotSlide);
                }
                let arme = piece.active_arme().ok_or(IllegalMove::PieceUnderAbriCannotSlide)?;
                let (dir, dist) = slide_dir(from, to).ok_or(IllegalMove::NotALine)?;
                let ok_dir = match arme {
                    Arme::Pierre => true,
                    Arme::Feuille => !dir.is_diagonal(),
                    Arme::Ciseaux => dir.is_diagonal(),
                };
                if !ok_dir {
                    return Err(IllegalMove::WrongDirectionForArme);
                }
                if piece.kind == Kind::Capitaine && dist != 1 {
                    return Err(IllegalMove::DistanceTooFarForCapitaine);
                }
                let (df, dr) = dir.delta();
                let mut cur = from;
                for _ in 0..dist.saturating_sub(1) {
                    cur = cur.offset(df, dr).ok_or(IllegalMove::OffBoard)?;
                    if self.at(cur).is_some() {
                        return Err(IllegalMove::PathBlocked);
                    }
                }
                let dest = cur.offset(df, dr).ok_or(IllegalMove::OffBoard)?;
                debug_assert_eq!(dest, to);
                if let Some(other) = self.at(dest) {
                    if other.color == piece.color {
                        return Err(IllegalMove::FriendlyAtDestination);
                    }
                    if other.is_under_abri() {
                        return Err(IllegalMove::TargetUnderAbri);
                    }
                    let Some(b) = other.active_arme() else {
                        return Err(IllegalMove::TargetUnderAbri);
                    };
                    if !arme.beats(b) {
                        return Err(IllegalMove::CannotBeatTarget);
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply a move that is already known to be legal (e.g. produced by
    /// `legal_moves`). Skips validation and returns an `Undo` token that
    /// can be passed to `unmake` to restore the previous state. Used by
    /// search / perft hot paths.
    pub fn apply_legal(&mut self, mv: Move) -> Undo {
        let from = mv.from();
        let piece = *self.at(from).expect("apply_legal: empty from-square");
        let plies_pre = self.plies_since_progress;
        let mut new_piece = piece;
        let mut captured: Option<(Pos, Piece)> = None;
        let color_idx = piece.color as usize;
        let kind_idx = piece.kind as usize;

        match mv {
            Move::Pivot { rot, .. } => {
                new_piece.cube = piece.cube.pivot(rot);
                self.update_streak(&mut new_piece, MoveKind::Pivot);
                if new_piece.streak >= 3 {
                    self.take(from);
                    self.captured.push(new_piece);
                    self.alive[color_idx][kind_idx] -= 1;
                } else {
                    // In-place: piece stays at `from`, bitboard bit stays set.
                    // We bypass set/take so we must flip the Zobrist contribution
                    // manually: out with the old piece-state, in with the new.
                    self.zobrist ^= piece_key(from, piece) ^ piece_key(from, new_piece);
                    *self.at_mut(from) = Some(new_piece);
                }
                self.plies_since_progress += 1;
            }
            Move::Bascule { dir, .. } => {
                let (df, dr) = dir.delta();
                let target = from.offset(df, dr).expect("bascule off board");
                new_piece.cube = piece.cube.bascule(dir);
                self.take(from);
                self.update_streak(&mut new_piece, MoveKind::Bascule);
                if new_piece.streak >= 3 {
                    self.captured.push(new_piece);
                    self.alive[color_idx][kind_idx] -= 1;
                } else {
                    self.set(target, new_piece);
                }
                self.plies_since_progress += 1;
            }
            Move::Deplacement { to, .. } => {
                if let Some(c) = self.take(to) {
                    self.alive[c.color as usize][c.kind as usize] -= 1;
                    captured = Some((to, c));
                }
                self.take(from);
                self.update_streak(&mut new_piece, MoveKind::Deplacement);
                if new_piece.streak >= 3 {
                    self.captured.push(new_piece);
                    self.alive[color_idx][kind_idx] -= 1;
                } else {
                    self.set(to, new_piece);
                }
                self.plies_since_progress = 0;
            }
        }

        let self_removed = new_piece.streak >= 3;
        if let Some((_, c)) = captured {
            // pushed AFTER any self_removed push so unmake can pop in reverse.
            self.captured.push(c);
        }

        // Update Zobrist for side/plies changes. The piece keys were already
        // XORed in/out by set/take (or by the pivot-in-place fast path).
        if plies_pre != self.plies_since_progress {
            self.zobrist ^= plies_key(plies_pre) ^ plies_key(self.plies_since_progress);
        }
        self.zobrist ^= ZOB_SIDE_TO_MOVE;

        self.ply += 1;
        self.side_to_move = self.side_to_move.opp();

        Undo {
            moved_piece_pre: piece,
            captured,
            self_removed,
            plies_since_progress_pre: plies_pre,
        }
    }

    /// Reverse the effect of `apply_legal(mv)`.
    /// `apply_legal` pushes onto `captured` in order [self_removed?, captured?];
    /// this method pops in reverse.
    pub fn unmake(&mut self, mv: Move, undo: Undo) {
        self.ply -= 1;
        self.side_to_move = self.side_to_move.opp();
        let plies_post = self.plies_since_progress;
        self.plies_since_progress = undo.plies_since_progress_pre;
        // Mirror of the side+plies XORs in apply_legal.
        if plies_post != undo.plies_since_progress_pre {
            self.zobrist ^= plies_key(plies_post) ^ plies_key(undo.plies_since_progress_pre);
        }
        self.zobrist ^= ZOB_SIDE_TO_MOVE;
        let from = mv.from();
        let pre = undo.moved_piece_pre;

        // Pop captured first (it was the last push), then self-removed.
        if let Some((_, c)) = undo.captured {
            self.captured.pop();
            self.alive[c.color as usize][c.kind as usize] += 1;
        }
        if undo.self_removed {
            self.captured.pop();
            self.alive[pre.color as usize][pre.kind as usize] += 1;
        }

        match mv {
            Move::Pivot { .. } => {
                if undo.self_removed {
                    // Square was emptied during apply; restore.
                    self.set(from, pre);
                } else {
                    // Piece stayed at `from` with rotated cube; overwrite in
                    // place. Bitboard bit was never cleared. Flip Zobrist
                    // contribution to undo the in-place piece change.
                    let rotated = self.at(from).copied().expect("pivot piece missing");
                    self.zobrist ^= piece_key(from, rotated) ^ piece_key(from, pre);
                    *self.at_mut(from) = Some(pre);
                }
            }
            Move::Bascule { dir, .. } => {
                let (df, dr) = dir.delta();
                let target = from.offset(df, dr).expect("bascule off board");
                if !undo.self_removed {
                    self.take(target);
                }
                self.set(from, pre);
            }
            Move::Deplacement { to, .. } => {
                if !undo.self_removed {
                    self.take(to);
                }
                if let Some((_, cpiece)) = undo.captured {
                    self.set(to, cpiece);
                }
                self.set(from, pre);
            }
        }
    }

    fn update_streak(&mut self, piece: &mut Piece, kind: MoveKind) {
        if piece.last_kind == Some(kind) {
            piece.streak = piece.streak.saturating_add(1);
        } else {
            piece.last_kind = Some(kind);
            piece.streak = 1;
        }
    }

}

