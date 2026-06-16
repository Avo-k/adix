use crate::board::Board;
use crate::geom::{Dir4, Pos, RotDir};
use crate::moves::Move;
use crate::piece::{Color, Kind};

pub fn parse_pos(s: &str) -> Option<Pos> {
    let b = s.as_bytes();
    if b.len() != 2 { return None; }
    let f = b[0].to_ascii_lowercase();
    let r = b[1];
    if !(b'a'..=b'i').contains(&f) { return None; }
    if !(b'1'..=b'9').contains(&r) { return None; }
    Some(Pos::new(f - b'a', r - b'1'))
}

pub fn parse_dir4(c: char) -> Option<Dir4> {
    match c.to_ascii_lowercase() {
        'n' => Some(Dir4::N),
        's' => Some(Dir4::S),
        'e' => Some(Dir4::E),
        'w' => Some(Dir4::W),
        _ => None,
    }
}

pub fn parse_rot(c: char) -> Option<RotDir> {
    match c.to_ascii_lowercase() {
        'l' => Some(RotDir::Left),
        'r' => Some(RotDir::Right),
        _ => None,
    }
}

/// Accepted forms:
///   e1-e2       deplacement
///   e1>n        bascule (n/s/e/w)
///   e1@l        pivot   (l/r)
pub fn parse_move(s: &str) -> Option<Move> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() == 5 && bytes[2] == b'-' {
        let from = parse_pos(&s[0..2])?;
        let to = parse_pos(&s[3..5])?;
        return Some(Move::Deplacement { from, to });
    }
    if bytes.len() == 4 && bytes[2] == b'>' {
        let from = parse_pos(&s[0..2])?;
        let dir = parse_dir4(s.as_bytes()[3] as char)?;
        return Some(Move::Bascule { from, dir });
    }
    if bytes.len() == 4 && bytes[2] == b'@' {
        let from = parse_pos(&s[0..2])?;
        let rot = parse_rot(s.as_bytes()[3] as char)?;
        return Some(Move::Pivot { from, rot });
    }
    None
}

pub fn fmt_move(m: Move) -> String {
    match m {
        Move::Deplacement { from, to } => format!("{}-{}", from, to),
        Move::Bascule { from, dir } => format!("{}>{}", from, dir),
        Move::Pivot { from, rot } => {
            let r = match rot { RotDir::Left => 'l', RotDir::Right => 'r' };
            format!("{}@{}", from, r)
        }
    }
}

/// ASCII board renderer. Each cell is 3 chars wide. Top of output is rank 9 (north).
pub fn render(b: &Board) -> String {
    let mut s = String::new();
    s.push_str("    a  b  c  d  e  f  g  h  i\n");
    s.push_str("   +--+--+--+--+--+--+--+--+--+\n");
    for rank in (0..9).rev() {
        s.push_str(&format!(" {} |", rank + 1));
        for file in 0..9 {
            let pos = Pos::new(file, rank as u8);
            let cell = match b.at(pos) {
                None => {
                    if pos.is_dark() { "## ".to_string() } else { "   ".to_string() }
                }
                Some(p) => {
                    let face = p.cube.active().glyph();
                    let color_marker = match p.color {
                        Color::Clair => 'w',
                        Color::Fonce => 'b',
                    };
                    let kind_marker = match p.kind {
                        Kind::Capitaine => '*',
                        Kind::Equipier => ' ',
                    };
                    format!("{}{}{}", color_marker, face, kind_marker)
                }
            };
            s.push_str(&cell);
            s.push('|');
        }
        s.push_str(&format!(" {}\n", rank + 1));
        s.push_str("   +--+--+--+--+--+--+--+--+--+\n");
    }
    s.push_str("    a  b  c  d  e  f  g  h  i\n");
    s.push_str(&format!(
        "side to move: {}   ply: {}   draw counter: {}/{}   captured: {}\n",
        match b.side_to_move { Color::Clair => "white (clair)", Color::Fonce => "black (foncé)" },
        b.ply,
        b.plies_since_progress,
        crate::board::DRAW_PLY_LIMIT,
        b.captured.len(),
    ));
    s
}
