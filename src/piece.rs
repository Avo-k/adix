use crate::geom::{Dir4, RotDir};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Arme {
    Pierre,
    Feuille,
    Ciseaux,
}

impl Arme {
    /// pierre > ciseaux > feuille > pierre.
    pub fn beats(self, other: Arme) -> bool {
        matches!(
            (self, other),
            (Arme::Pierre, Arme::Ciseaux)
                | (Arme::Ciseaux, Arme::Feuille)
                | (Arme::Feuille, Arme::Pierre)
        )
    }
    pub fn glyph(self) -> char {
        match self {
            Arme::Pierre => 'O',
            Arme::Feuille => '+',
            Arme::Ciseaux => 'X',
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Face {
    Arme(Arme),
    Abri,
}

impl Face {
    pub fn glyph(self) -> char {
        match self {
            Face::Arme(a) => a.glyph(),
            Face::Abri => '^',
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Color {
    Clair,
    Fonce,
}

impl Color {
    pub fn opp(self) -> Color {
        match self {
            Color::Clair => Color::Fonce,
            Color::Fonce => Color::Clair,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Kind {
    Capitaine,
    Equipier,
}

/// Six oriented faces of a cube. Axes: +y = N, +x = E, +z = up.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Cube {
    pub top: Face,
    pub bottom: Face,
    pub north: Face,
    pub south: Face,
    pub east: Face,
    pub west: Face,
}

impl Cube {
    pub fn active(&self) -> Face {
        self.top
    }

    /// Tumble ¼ turn over the edge in direction `d`.
    pub fn bascule(self, d: Dir4) -> Cube {
        let Cube { top, bottom, north, south, east, west } = self;
        match d {
            Dir4::N => Cube { top: south, bottom: north, north: top, south: bottom, east, west },
            Dir4::S => Cube { top: north, bottom: south, north: bottom, south: top, east, west },
            Dir4::E => Cube { top: west, bottom: east, east: top, west: bottom, north, south },
            Dir4::W => Cube { top: east, bottom: west, east: bottom, west: top, north, south },
        }
    }

    /// Rotate ¼ turn around the vertical axis.
    /// Right = clockwise viewed from above.
    pub fn pivot(self, r: RotDir) -> Cube {
        let Cube { top, bottom, north, south, east, west } = self;
        match r {
            RotDir::Right => Cube { top, bottom, north: east, east: south, south: west, west: north },
            RotDir::Left => Cube { top, bottom, north: west, west: south, south: east, east: north },
        }
    }
}

/// Starting orientation for both colors per §6:
/// - Capitaine: pierre on top, feuille N/S, ciseaux E/W
/// - Equipier:  abri on top, pierre on bottom, feuille N/S, ciseaux E/W
pub fn starting_capitaine_cube() -> Cube {
    Cube {
        top: Face::Arme(Arme::Pierre),
        bottom: Face::Arme(Arme::Pierre),
        north: Face::Arme(Arme::Feuille),
        south: Face::Arme(Arme::Feuille),
        east: Face::Arme(Arme::Ciseaux),
        west: Face::Arme(Arme::Ciseaux),
    }
}

pub fn starting_equipier_cube() -> Cube {
    Cube {
        top: Face::Abri,
        bottom: Face::Arme(Arme::Pierre),
        north: Face::Arme(Arme::Feuille),
        south: Face::Arme(Arme::Feuille),
        east: Face::Arme(Arme::Ciseaux),
        west: Face::Arme(Arme::Ciseaux),
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum MoveKind {
    Deplacement,
    Bascule,
    Pivot,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Piece {
    pub color: Color,
    pub kind: Kind,
    pub cube: Cube,
    pub last_kind: Option<MoveKind>,
    pub streak: u8,
}

impl Piece {
    pub fn new(color: Color, kind: Kind, cube: Cube) -> Self {
        Self { color, kind, cube, last_kind: None, streak: 0 }
    }
    pub fn active_arme(&self) -> Option<Arme> {
        match self.cube.active() {
            Face::Arme(a) => Some(a),
            Face::Abri => None,
        }
    }
    pub fn is_under_abri(&self) -> bool {
        matches!(self.cube.active(), Face::Abri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_cube() -> Cube {
        Cube {
            top: Face::Arme(Arme::Pierre),
            bottom: Face::Arme(Arme::Feuille),
            north: Face::Arme(Arme::Ciseaux),
            south: Face::Abri,
            east: Face::Arme(Arme::Pierre),
            west: Face::Arme(Arme::Feuille),
        }
    }

    #[test]
    fn bascule_4_is_identity() {
        for d in Dir4::ALL {
            let c = diag_cube();
            let after = c.bascule(d).bascule(d).bascule(d).bascule(d);
            assert_eq!(after, c, "bascule {:?} ×4 should be identity", d);
        }
    }

    #[test]
    fn pivot_4_is_identity() {
        for r in [RotDir::Left, RotDir::Right] {
            let c = diag_cube();
            let after = c.pivot(r).pivot(r).pivot(r).pivot(r);
            assert_eq!(after, c, "pivot {:?} ×4 should be identity", r);
        }
    }

    #[test]
    fn bascule_then_opposite_is_identity() {
        let c = diag_cube();
        assert_eq!(c.bascule(Dir4::N).bascule(Dir4::S), c);
        assert_eq!(c.bascule(Dir4::E).bascule(Dir4::W), c);
    }

    #[test]
    fn pivot_l_then_r_is_identity() {
        let c = diag_cube();
        assert_eq!(c.pivot(RotDir::Left).pivot(RotDir::Right), c);
    }

    #[test]
    fn rps_cycle() {
        assert!(Arme::Pierre.beats(Arme::Ciseaux));
        assert!(Arme::Ciseaux.beats(Arme::Feuille));
        assert!(Arme::Feuille.beats(Arme::Pierre));
        assert!(!Arme::Pierre.beats(Arme::Feuille));
        assert!(!Arme::Pierre.beats(Arme::Pierre));
    }

    #[test]
    fn equipier_bascule_n_brings_feuille_up() {
        // bascule(N): new top = old south. Equipier's south face is feuille, so the
        // equipier leaving its abri toward the north now shows feuille on top.
        let c = starting_equipier_cube();
        let after = c.bascule(Dir4::N);
        assert_eq!(after.top, Face::Arme(Arme::Feuille));
        // new north = old top (abri).
        assert_eq!(after.north, Face::Abri);
        // new bottom = old north (feuille).
        assert_eq!(after.bottom, Face::Arme(Arme::Feuille));
        // new south = old bottom (pierre).
        assert_eq!(after.south, Face::Arme(Arme::Pierre));
    }
}
