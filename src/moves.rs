use crate::geom::{Dir4, Dir8, Pos, RotDir};
use crate::piece::MoveKind;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Move {
    Deplacement { from: Pos, to: Pos },
    Bascule { from: Pos, dir: Dir4 },
    Pivot { from: Pos, rot: RotDir },
}

impl Move {
    pub fn from(self) -> Pos {
        match self {
            Move::Deplacement { from, .. } | Move::Bascule { from, .. } | Move::Pivot { from, .. } => from,
        }
    }
    pub fn kind(self) -> MoveKind {
        match self {
            Move::Deplacement { .. } => MoveKind::Deplacement,
            Move::Bascule { .. } => MoveKind::Bascule,
            Move::Pivot { .. } => MoveKind::Pivot,
        }
    }
}

/// Direction of a sliding deplacement from `from` to `to`.
/// Returns `None` if (to - from) isn't on a single ortho or diagonal line.
pub fn slide_dir(from: Pos, to: Pos) -> Option<(Dir8, u8)> {
    let df = to.file as i8 - from.file as i8;
    let dr = to.rank as i8 - from.rank as i8;
    if df == 0 && dr == 0 {
        return None;
    }
    let adf = df.unsigned_abs();
    let adr = dr.unsigned_abs();
    let (dir, dist) = if df == 0 {
        (if dr > 0 { Dir8::N } else { Dir8::S }, adr)
    } else if dr == 0 {
        (if df > 0 { Dir8::E } else { Dir8::W }, adf)
    } else if adf == adr {
        let dir = match (df.signum(), dr.signum()) {
            (1, 1) => Dir8::NE,
            (-1, 1) => Dir8::NW,
            (1, -1) => Dir8::SE,
            (-1, -1) => Dir8::SW,
            _ => unreachable!(),
        };
        (dir, adf)
    } else {
        return None;
    };
    Some((dir, dist))
}
