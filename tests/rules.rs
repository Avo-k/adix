use adix::board::{Board, DRAW_PLY_LIMIT, Outcome};
use adix::geom::{Dir4, Pos, RotDir};
use adix::moves::Move;
use adix::notation::parse_pos;
use adix::piece::{Arme, Color, Cube, Face, Kind, Piece};

fn pos(s: &str) -> Pos {
    parse_pos(s).unwrap()
}

fn place_pierre(b: &mut Board, sq: &str, color: Color, kind: Kind) {
    let face = Face::Arme(Arme::Pierre);
    let cube = Cube { top: face, bottom: face, north: face, south: face, east: face, west: face };
    b.force_place(pos(sq), Piece::new(color, kind, cube));
}

#[test]
fn three_consecutive_basculs_remove_piece_even_with_mixed_dirs() {
    let mut b = Board::empty_for_test();
    place_pierre(&mut b, "e5", Color::Clair, Kind::Equipier);
    place_pierre(&mut b, "i9", Color::Fonce, Kind::Equipier); // black filler so game doesn't end
    place_pierre(&mut b, "a1", Color::Fonce, Kind::Capitaine);
    place_pierre(&mut b, "a9", Color::Clair, Kind::Capitaine);
    b.set_side_to_move(Color::Clair);

    // White basculsthrice in different directions, with black moving in between.
    b.apply(Move::Bascule { from: pos("e5"), dir: Dir4::N }).unwrap();      // white: e5→e6
    b.apply(Move::Pivot { from: pos("i9"), rot: RotDir::Left }).unwrap();   // black filler
    b.apply(Move::Bascule { from: pos("e6"), dir: Dir4::E }).unwrap();      // white: e6→f6
    b.apply(Move::Pivot { from: pos("i9"), rot: RotDir::Right }).unwrap();  // black filler
    // 3rd bascule should remove the white equipier.
    b.apply(Move::Bascule { from: pos("f6"), dir: Dir4::S }).unwrap();      // white: f6→? but removed
    assert!(b.at(pos("f6")).is_none(), "piece should have been removed after 3rd bascule");
    assert!(b.at(pos("f5")).is_none(), "removed piece is not placed at destination either");
}

#[test]
fn pivot_breaks_bascule_streak() {
    let mut b = Board::empty_for_test();
    place_pierre(&mut b, "e5", Color::Clair, Kind::Equipier);
    place_pierre(&mut b, "i9", Color::Fonce, Kind::Equipier);
    place_pierre(&mut b, "a1", Color::Fonce, Kind::Capitaine);
    place_pierre(&mut b, "a9", Color::Clair, Kind::Capitaine);
    b.set_side_to_move(Color::Clair);

    b.apply(Move::Bascule { from: pos("e5"), dir: Dir4::N }).unwrap();      // white: bascule (streak=1)
    b.apply(Move::Pivot { from: pos("i9"), rot: RotDir::Left }).unwrap();   // black filler
    b.apply(Move::Pivot { from: pos("e6"), rot: RotDir::Left }).unwrap();   // white: pivot resets kind
    b.apply(Move::Pivot { from: pos("i9"), rot: RotDir::Right }).unwrap();  // black filler
    b.apply(Move::Bascule { from: pos("e6"), dir: Dir4::S }).unwrap();      // white: bascule (streak=1 again)
    b.apply(Move::Pivot { from: pos("i9"), rot: RotDir::Left }).unwrap();   // black filler
    b.apply(Move::Bascule { from: pos("e5"), dir: Dir4::N }).unwrap();      // white: bascule (streak=2)
    assert!(b.at(pos("e6")).is_some(), "piece must still be on board: streak only 2 after pivot reset");
}

#[test]
fn capturing_capitaine_wins() {
    let mut b = Board::empty_for_test();
    place_pierre(&mut b, "a1", Color::Clair, Kind::Equipier);
    // black capitaine on pierre at a9 — white equipier with pierre cannot beat pierre.
    // Let's give black ciseaux (which pierre beats).
    let ciseaux = Face::Arme(Arme::Ciseaux);
    let cube = Cube { top: ciseaux, bottom: ciseaux, north: ciseaux, south: ciseaux, east: ciseaux, west: ciseaux };
    b.force_place(pos("a9"), Piece::new(Color::Fonce, Kind::Capitaine, cube));
    place_pierre(&mut b, "i1", Color::Fonce, Kind::Equipier);
    place_pierre(&mut b, "i9", Color::Clair, Kind::Capitaine);
    b.set_side_to_move(Color::Clair);

    let outcome = b.apply(Move::Deplacement { from: pos("a1"), to: pos("a9") }).unwrap();
    assert_eq!(outcome, Some(Outcome::Win(Color::Clair)));
}

#[test]
fn eliminating_all_equipiers_wins() {
    // White capitaine + 1 white equipier vs black capitaine alone (0 equipiers) → already won? No,
    // the win triggers at the moment the *last* equipier is captured. We simulate the capture.
    let mut b = Board::empty_for_test();
    place_pierre(&mut b, "i1", Color::Clair, Kind::Capitaine); // white cap
    place_pierre(&mut b, "i9", Color::Fonce, Kind::Capitaine); // black cap
    // Black's last equipier on a9 with ciseaux; white's pierre eq on a1 will capture it.
    place_pierre(&mut b, "a1", Color::Clair, Kind::Equipier);
    let ciseaux = Face::Arme(Arme::Ciseaux);
    let cube = Cube { top: ciseaux, bottom: ciseaux, north: ciseaux, south: ciseaux, east: ciseaux, west: ciseaux };
    b.force_place(pos("a9"), Piece::new(Color::Fonce, Kind::Equipier, cube));
    b.set_side_to_move(Color::Clair);

    let outcome = b.apply(Move::Deplacement { from: pos("a1"), to: pos("a9") }).unwrap();
    assert_eq!(outcome, Some(Outcome::Win(Color::Clair)));
}

#[test]
fn draw_counter_at_limit_yields_draw() {
    let mut b = Board::empty_for_test();
    place_pierre(&mut b, "a1", Color::Clair, Kind::Capitaine);
    place_pierre(&mut b, "a9", Color::Fonce, Kind::Capitaine);
    place_pierre(&mut b, "i1", Color::Clair, Kind::Equipier);
    place_pierre(&mut b, "i9", Color::Fonce, Kind::Equipier);
    assert!(b.outcome().is_none());
    b.plies_since_progress = DRAW_PLY_LIMIT;
    assert_eq!(b.outcome(), Some(Outcome::Draw));
}

#[test]
fn pivot_increments_counter_then_deplacement_resets() {
    let mut b = Board::empty_for_test();
    place_pierre(&mut b, "a1", Color::Clair, Kind::Capitaine);
    place_pierre(&mut b, "a9", Color::Fonce, Kind::Capitaine);
    place_pierre(&mut b, "e5", Color::Clair, Kind::Equipier);
    place_pierre(&mut b, "e9", Color::Fonce, Kind::Equipier); // off the slide path
    b.set_side_to_move(Color::Clair);

    b.apply(Move::Pivot { from: pos("e5"), rot: RotDir::Left }).unwrap();
    assert_eq!(b.plies_since_progress, 1);
    b.apply(Move::Pivot { from: pos("e9"), rot: RotDir::Right }).unwrap();
    assert_eq!(b.plies_since_progress, 2);
    b.apply(Move::Deplacement { from: pos("e5"), to: pos("e6") }).unwrap();
    assert_eq!(b.plies_since_progress, 0);
}
