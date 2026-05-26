use crate::geom::{Dir4, Dir8, Pos, RotDir};
use crate::moves::{Move, slide_dir};
use crate::piece::{
    Arme, Color, Kind, MoveKind, Piece, starting_capitaine_cube, starting_equipier_cube,
};

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

#[derive(Clone, Debug)]
pub struct Board {
    cells: [[Option<Piece>; SIZE]; SIZE],
    pub side_to_move: Color,
    pub plies_since_progress: u32,
    pub ply: u32,
    pub captured: Vec<Piece>,
}

impl Board {
    pub fn empty() -> Self {
        Self {
            cells: Default::default(),
            side_to_move: Color::Clair,
            plies_since_progress: 0,
            ply: 0,
            captured: Vec::new(),
        }
    }

    /// Empty board used by tests to set up specific positions.
    pub fn empty_for_test() -> Self {
        Self::empty()
    }

    pub fn force_place(&mut self, p: Pos, piece: Piece) {
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
        b.set(white_cap, Piece::new(Color::Clair, Kind::Capitaine, cap));
        for p in white_eqs {
            b.set(p, Piece::new(Color::Clair, Kind::Equipier, eq));
        }

        // Black (foncé) at north.
        let black_cap = Pos::new(4, 8); // e9
        let black_eqs: [Pos; 9] = [
            Pos::new(2, 8), Pos::new(6, 8),                                    // c9 g9
            Pos::new(1, 7), Pos::new(3, 7), Pos::new(5, 7), Pos::new(7, 7),    // b8 d8 f8 h8
            Pos::new(2, 6), Pos::new(4, 6), Pos::new(6, 6),                    // c7 e7 g7
        ];
        b.set(black_cap, Piece::new(Color::Fonce, Kind::Capitaine, cap));
        for p in black_eqs {
            b.set(p, Piece::new(Color::Fonce, Kind::Equipier, eq));
        }

        b
    }

    pub fn at(&self, p: Pos) -> Option<&Piece> {
        self.cells[p.rank as usize][p.file as usize].as_ref()
    }
    fn at_mut(&mut self, p: Pos) -> &mut Option<Piece> {
        &mut self.cells[p.rank as usize][p.file as usize]
    }
    fn set(&mut self, p: Pos, piece: Piece) {
        *self.at_mut(p) = Some(piece);
    }
    fn take(&mut self, p: Pos) -> Option<Piece> {
        self.at_mut(p).take()
    }

    pub fn iter_pieces(&self) -> impl Iterator<Item = (Pos, &Piece)> {
        (0..SIZE).flat_map(move |r| {
            (0..SIZE).filter_map(move |f| {
                let p = Pos::new(f as u8, r as u8);
                self.at(p).map(|piece| (p, piece))
            })
        })
    }

    /// Counts (capitaines_alive, equipiers_alive) for the given color.
    pub fn alive_counts(&self, color: Color) -> (u32, u32) {
        let mut cap = 0;
        let mut eq = 0;
        for (_, p) in self.iter_pieces() {
            if p.color != color { continue; }
            match p.kind {
                Kind::Capitaine => cap += 1,
                Kind::Equipier => eq += 1,
            }
        }
        (cap, eq)
    }

    pub fn outcome(&self) -> Option<Outcome> {
        let (wc, we) = self.alive_counts(Color::Clair);
        let (bc, be) = self.alive_counts(Color::Fonce);
        if wc == 0 || we == 0 {
            return Some(Outcome::Win(Color::Fonce));
        }
        if bc == 0 || be == 0 {
            return Some(Outcome::Win(Color::Clair));
        }
        if self.plies_since_progress >= DRAW_PLY_LIMIT {
            return Some(Outcome::Draw);
        }
        None
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::new();
        for (pos, piece) in self.iter_pieces() {
            if piece.color != self.side_to_move {
                continue;
            }
            self.gen_moves_from(pos, piece, &mut out);
        }
        out
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
        let from = mv.from();
        let piece = self.at(from).copied().ok_or(IllegalMove::NoPieceThere)?;
        if piece.color != self.side_to_move {
            return Err(IllegalMove::NotYourTurn);
        }

        // Validate and execute.
        let mut captured_piece: Option<Piece> = None;
        let mut new_piece = piece;

        match mv {
            Move::Pivot { rot, .. } => {
                if piece.is_under_abri() {
                    return Err(IllegalMove::PieceUnderAbriCannotPivot);
                }
                new_piece.cube = piece.cube.pivot(rot);
            }
            Move::Bascule { dir, .. } => {
                let (df, dr) = dir.delta();
                let target = from.offset(df, dr).ok_or(IllegalMove::BasculeOffBoard)?;
                if self.at(target).is_some() {
                    return Err(IllegalMove::BasculeOntoOccupied);
                }
                new_piece.cube = piece.cube.bascule(dir);
                // move piece
                self.take(from);
                // re-place at target after streak/3x is applied below; defer to after match
                // We use a sentinel: store target for later move.
                self.update_streak(&mut new_piece, MoveKind::Bascule);
                self.commit_move(target, new_piece);
                self.post_move(MoveKind::Bascule, captured_piece.is_some());
                return Ok(self.outcome());
            }
            Move::Deplacement { from: _, to } => {
                if piece.is_under_abri() {
                    return Err(IllegalMove::PieceUnderAbriCannotSlide);
                }
                let arme = piece.active_arme().ok_or(IllegalMove::PieceUnderAbriCannotSlide)?;
                let (dir, dist) = slide_dir(from, to).ok_or(IllegalMove::NotALine)?;
                // direction must match arme
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
                // walk path: every intermediate square must be empty
                let (df, dr) = dir.delta();
                let mut cur = from;
                for _ in 0..dist.saturating_sub(1) {
                    cur = cur.offset(df, dr).ok_or(IllegalMove::OffBoard)?;
                    if self.at(cur).is_some() {
                        return Err(IllegalMove::PathBlocked);
                    }
                }
                // destination
                let dest = cur.offset(df, dr).ok_or(IllegalMove::OffBoard)?;
                debug_assert_eq!(dest, to);
                match self.at(dest) {
                    None => {} // empty: simple move
                    Some(other) => {
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
                        captured_piece = self.take(dest);
                    }
                }
                // execute
                self.take(from);
                // cube unchanged for deplacement (§8-2 "conserve l'orientation")
                self.update_streak(&mut new_piece, MoveKind::Deplacement);
                self.commit_move(dest, new_piece);
                if let Some(captured) = captured_piece.take() {
                    self.captured.push(captured);
                }
                self.post_move(MoveKind::Deplacement, true);
                return Ok(self.outcome());
            }
        }

        // Pivot fall-through: cube changed but piece stayed put.
        self.update_streak(&mut new_piece, MoveKind::Pivot);
        self.commit_move(from, new_piece);
        self.post_move(MoveKind::Pivot, false);
        Ok(self.outcome())
    }

    fn update_streak(&mut self, piece: &mut Piece, kind: MoveKind) {
        if piece.last_kind == Some(kind) {
            piece.streak = piece.streak.saturating_add(1);
        } else {
            piece.last_kind = Some(kind);
            piece.streak = 1;
        }
    }

    /// Place `piece` at `pos`, unless its streak hit 3 — in which case it self-removes.
    fn commit_move(&mut self, pos: Pos, piece: Piece) {
        if piece.streak >= 3 {
            self.captured.push(piece);
        } else {
            *self.at_mut(pos) = Some(piece);
        }
    }

    fn post_move(&mut self, kind: MoveKind, was_displacement_or_capture: bool) {
        let resets = matches!(kind, MoveKind::Deplacement) || was_displacement_or_capture;
        if resets {
            self.plies_since_progress = 0;
        } else {
            self.plies_since_progress += 1;
        }
        self.ply += 1;
        self.side_to_move = self.side_to_move.opp();
    }
}

