use adix::board::Board;
use adix::geom::Pos;
use adix::notation::parse_pos;
use adix::piece::{Arme, Color, Face, Kind};

fn pos(s: &str) -> Pos {
    parse_pos(s).unwrap()
}

#[test]
fn white_pieces_match_spec() {
    let b = Board::initial();
    let cap = b.at(pos("e1")).expect("white cap on e1");
    assert_eq!(cap.color, Color::Clair);
    assert_eq!(cap.kind, Kind::Capitaine);
    assert_eq!(cap.cube.active(), Face::Arme(Arme::Pierre));

    let white_eqs = ["c1", "g1", "b2", "d2", "f2", "h2", "c3", "e3", "g3"];
    for sq in white_eqs {
        let p = b.at(pos(sq)).unwrap_or_else(|| panic!("white equipier missing at {}", sq));
        assert_eq!(p.color, Color::Clair);
        assert_eq!(p.kind, Kind::Equipier);
        assert_eq!(p.cube.active(), Face::Abri);
    }
}

#[test]
fn black_pieces_match_spec() {
    let b = Board::initial();
    let cap = b.at(pos("e9")).expect("black cap on e9");
    assert_eq!(cap.color, Color::Fonce);
    assert_eq!(cap.kind, Kind::Capitaine);

    let black_eqs = ["c9", "g9", "b8", "d8", "f8", "h8", "c7", "e7", "g7"];
    for sq in black_eqs {
        let p = b.at(pos(sq)).unwrap_or_else(|| panic!("black equipier missing at {}", sq));
        assert_eq!(p.color, Color::Fonce);
        assert_eq!(p.kind, Kind::Equipier);
        assert_eq!(p.cube.active(), Face::Abri);
    }
}

#[test]
fn counts() {
    let b = Board::initial();
    let total = b.iter_pieces().count();
    assert_eq!(total, 20);
    assert_eq!(b.alive_counts(Color::Clair), (1, 9));
    assert_eq!(b.alive_counts(Color::Fonce), (1, 9));
}

#[test]
fn white_moves_first() {
    let b = Board::initial();
    assert_eq!(b.side_to_move, Color::Clair);
}

#[test]
fn ply_and_draw_counter_start_at_zero() {
    let b = Board::initial();
    assert_eq!(b.ply, 0);
    assert_eq!(b.plies_since_progress, 0);
    assert!(b.outcome().is_none());
}
