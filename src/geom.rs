use std::fmt;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Pos {
    pub file: u8,
    pub rank: u8,
}

impl Pos {
    pub const fn new(file: u8, rank: u8) -> Self {
        Self { file, rank }
    }

    pub fn in_bounds(file: i8, rank: i8) -> bool {
        (0..9).contains(&file) && (0..9).contains(&rank)
    }

    pub fn offset(self, df: i8, dr: i8) -> Option<Pos> {
        let f = self.file as i8 + df;
        let r = self.rank as i8 + dr;
        if Self::in_bounds(f, r) {
            Some(Pos::new(f as u8, r as u8))
        } else {
            None
        }
    }

    /// Dark squares: (file + rank) even — a1 (0,0) is dark.
    pub fn is_dark(self) -> bool {
        (self.file + self.rank) % 2 == 0
    }
}

impl fmt::Display for Pos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", (b'a' + self.file) as char, self.rank + 1)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Dir4 {
    N,
    S,
    E,
    W,
}

impl Dir4 {
    pub fn delta(self) -> (i8, i8) {
        match self {
            Dir4::N => (0, 1),
            Dir4::S => (0, -1),
            Dir4::E => (1, 0),
            Dir4::W => (-1, 0),
        }
    }
    pub const ALL: [Dir4; 4] = [Dir4::N, Dir4::S, Dir4::E, Dir4::W];
}

impl fmt::Display for Dir4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Dir4::N => "n",
            Dir4::S => "s",
            Dir4::E => "e",
            Dir4::W => "w",
        };
        f.write_str(s)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Dir8 {
    N,
    S,
    E,
    W,
    NE,
    NW,
    SE,
    SW,
}

impl Dir8 {
    pub fn delta(self) -> (i8, i8) {
        match self {
            Dir8::N => (0, 1),
            Dir8::S => (0, -1),
            Dir8::E => (1, 0),
            Dir8::W => (-1, 0),
            Dir8::NE => (1, 1),
            Dir8::NW => (-1, 1),
            Dir8::SE => (1, -1),
            Dir8::SW => (-1, -1),
        }
    }
    pub fn is_diagonal(self) -> bool {
        matches!(self, Dir8::NE | Dir8::NW | Dir8::SE | Dir8::SW)
    }
    pub const ORTHO: [Dir8; 4] = [Dir8::N, Dir8::S, Dir8::E, Dir8::W];
    pub const DIAG: [Dir8; 4] = [Dir8::NE, Dir8::NW, Dir8::SE, Dir8::SW];
    pub const ALL: [Dir8; 8] = [
        Dir8::N, Dir8::S, Dir8::E, Dir8::W,
        Dir8::NE, Dir8::NW, Dir8::SE, Dir8::SW,
    ];
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum RotDir {
    Left,
    Right,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a1_is_dark() {
        assert!(Pos::new(0, 0).is_dark());
    }
    #[test]
    fn corners_dark() {
        for (f, r) in [(0, 0), (0, 8), (8, 0), (8, 8)] {
            assert!(Pos::new(f, r).is_dark(), "corner {f},{r} should be dark");
        }
    }
    #[test]
    fn display_pos() {
        assert_eq!(Pos::new(0, 0).to_string(), "a1");
        assert_eq!(Pos::new(4, 0).to_string(), "e1");
        assert_eq!(Pos::new(8, 8).to_string(), "i9");
    }
}
