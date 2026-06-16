//! Integration tests reproducing the official rule book's Annexe 3 capture
//! diagrams, plus general sliding and blocking behavior.

use adix::board::Board;
use adix::geom::Pos;
use adix::moves::Move;
use adix::notation::parse_pos;
use adix::piece::{Arme, Color, Cube, Face, Kind, Piece};

fn pos(s: &str) -> Pos {
    parse_pos(s).unwrap()
}

fn place(b: &mut Board, sq: &str, color: Color, kind: Kind, top: Face) {
    let cube = Cube { top, bottom: top, north: top, south: top, east: top, west: top };
    b.force_place(pos(sq), Piece::new(color, kind, cube));
}

#[test]
fn first_moves_for_white_in_initial_position() {
    let b = Board::initial();
    let moves = b.legal_moves();

    // White equipiers are all under abri: each can only bascule (no pivot, no deplacement).
    // White capitaine on e1 has pierre active: 1-square pierre moves + bascules + pivots.
    let from_e1: Vec<Move> = moves.iter().copied().filter(|m| m.from() == pos("e1")).collect();
    assert!(!from_e1.is_empty(), "capitaine on e1 should have legal moves");
    // capitaine bascule N is blocked by e2? e2 is empty in initial position. Verify.
    assert!(b.at(pos("e2")).is_none(), "e2 should be empty in initial");
    assert!(from_e1.contains(&Move::Bascule { from: pos("e1"), dir: adix::geom::Dir4::N }));

    // No piece should be allowed to pivot or slide from c1 (under abri).
    let from_c1: Vec<Move> = moves.iter().copied().filter(|m| m.from() == pos("c1")).collect();
    for m in &from_c1 {
        assert!(matches!(m, Move::Bascule { .. }), "abri piece should only bascule, got {:?}", m);
    }
}

#[test]
fn equipier_pierre_slides_unlimited_then_captures_per_annexe_3() {
    // Reproduce Annexe 3 case: "Pierre g3" captures "Ciseaux d6".
    // We set up an empty board and place exactly the two pieces.
    use adix::board::Board;
    let mut b = Board::empty_for_test();
    place(&mut b, "g3", Color::Clair, Kind::Equipier, Face::Arme(Arme::Pierre));
    place(&mut b, "d6", Color::Fonce, Kind::Equipier, Face::Arme(Arme::Ciseaux));
    b.set_side_to_move(Color::Clair);

    // g3 → d6 is NW direction, 3 squares; pierre beats ciseaux.
    let mv = Move::Deplacement { from: pos("g3"), to: pos("d6") };
    let res = b.apply(mv);
    assert!(res.is_ok(), "expected legal capture, got {:?}", res);
    assert!(b.at(pos("g3")).is_none());
    let cap = b.at(pos("d6")).expect("attacker should be at d6 now");
    assert_eq!(cap.color, Color::Clair);
    assert_eq!(b.captured.len(), 1);
}

#[test]
fn capture_blocked_by_intervening_piece() {
    // Annexe 3: "Ciseaux d6" cannot capture "Feuille f8" because abri at e7 blocks.
    use adix::board::Board;
    let mut b = Board::empty_for_test();
    place(&mut b, "d6", Color::Clair, Kind::Equipier, Face::Arme(Arme::Ciseaux));
    place(&mut b, "e7", Color::Fonce, Kind::Equipier, Face::Abri);
    place(&mut b, "f8", Color::Fonce, Kind::Equipier, Face::Arme(Arme::Feuille));
    b.set_side_to_move(Color::Clair);

    let mv = Move::Deplacement { from: pos("d6"), to: pos("f8") };
    let res = b.apply(mv);
    assert!(res.is_err(), "expected path-blocked, got {:?}", res);
}

#[test]
fn cannot_capture_abri() {
    // Annexe 3: nothing can capture "Abri e7".
    use adix::board::Board;
    let mut b = Board::empty_for_test();
    place(&mut b, "e7", Color::Fonce, Kind::Equipier, Face::Abri);
    place(&mut b, "e1", Color::Clair, Kind::Equipier, Face::Arme(Arme::Feuille));
    b.set_side_to_move(Color::Clair);
    let mv = Move::Deplacement { from: pos("e1"), to: pos("e7") };
    let res = b.apply(mv);
    assert!(res.is_err(), "expected target-under-abri, got {:?}", res);
}

#[test]
fn feuille_cannot_move_diagonally() {
    use adix::board::Board;
    let mut b = Board::empty_for_test();
    place(&mut b, "e5", Color::Clair, Kind::Equipier, Face::Arme(Arme::Feuille));
    b.set_side_to_move(Color::Clair);
    let mv = Move::Deplacement { from: pos("e5"), to: pos("f6") };
    let res = b.apply(mv);
    assert!(res.is_err());
}

#[test]
fn capitaine_only_one_square() {
    use adix::board::Board;
    let mut b = Board::empty_for_test();
    place(&mut b, "e5", Color::Clair, Kind::Capitaine, Face::Arme(Arme::Feuille));
    b.set_side_to_move(Color::Clair);
    // 1 square = ok
    let mut b1 = b.clone();
    assert!(b1.apply(Move::Deplacement { from: pos("e5"), to: pos("e6") }).is_ok());
    // 2 squares = illegal
    let mut b2 = b.clone();
    assert!(b2.apply(Move::Deplacement { from: pos("e5"), to: pos("e7") }).is_err());
}

#[test]
fn cannot_beat_same_arme() {
    use adix::board::Board;
    let mut b = Board::empty_for_test();
    place(&mut b, "a1", Color::Clair, Kind::Equipier, Face::Arme(Arme::Pierre));
    place(&mut b, "a9", Color::Fonce, Kind::Equipier, Face::Arme(Arme::Pierre));
    b.set_side_to_move(Color::Clair);
    let mv = Move::Deplacement { from: pos("a1"), to: pos("a9") };
    let res = b.apply(mv);
    assert!(res.is_err());
}
