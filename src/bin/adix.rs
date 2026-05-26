use std::io::{self, BufRead, Write};

use adix::board::{Board, Outcome};
use adix::notation::{fmt_move, parse_move, parse_pos, render};
use adix::piece::Color;

fn main() {
    let mut history: Vec<Board> = vec![Board::initial()];

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    println!("{}", render(history.last().unwrap()));
    print_help();

    loop {
        print!("adix> ");
        stdout.flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            println!();
            break;
        }
        let cmd = line.trim();
        if cmd.is_empty() {
            continue;
        }
        match cmd {
            "quit" | "exit" | "q" => break,
            "help" | "h" | "?" => print_help(),
            "board" | "b" => println!("{}", render(history.last().unwrap())),
            "moves" | "m" => {
                let b = history.last().unwrap();
                let ms = b.legal_moves();
                println!("{} legal moves:", ms.len());
                for chunk in ms.chunks(8) {
                    let line: Vec<String> = chunk.iter().map(|m| fmt_move(*m)).collect();
                    println!("  {}", line.join("  "));
                }
            }
            s if s.starts_with("moves ") => {
                let arg = s["moves ".len()..].trim();
                match parse_pos(arg) {
                    None => println!("? could not parse square '{}'", arg),
                    Some(p) => {
                        let ms = history.last().unwrap().legal_moves_from(p);
                        if ms.is_empty() {
                            println!("no legal moves from {}", p);
                        } else {
                            for m in &ms {
                                println!("  {}", fmt_move(*m));
                            }
                        }
                    }
                }
            }
            "undo" | "u" => {
                if history.len() > 1 {
                    history.pop();
                    println!("(undone)");
                    println!("{}", render(history.last().unwrap()));
                } else {
                    println!("nothing to undo");
                }
            }
            _ => match parse_move(cmd) {
                None => println!("? unknown command or move. type 'help'."),
                Some(mv) => {
                    let mut next = history.last().unwrap().clone();
                    match next.apply(mv) {
                        Err(e) => println!("illegal: {:?}", e),
                        Ok(maybe_outcome) => {
                            history.push(next);
                            println!("{}", render(history.last().unwrap()));
                            if let Some(o) = maybe_outcome {
                                match o {
                                    Outcome::Win(Color::Clair) => println!("** white (clair) wins **"),
                                    Outcome::Win(Color::Fonce) => println!("** black (foncé) wins **"),
                                    Outcome::Draw => println!("** draw **"),
                                }
                            }
                        }
                    }
                }
            },
        }
    }
}

fn print_help() {
    println!(
        "Commands:
  help, h, ?     this message
  board, b       re-print the board
  moves, m       list every legal move for the side to move
  moves <sq>     list legal moves from <sq>, e.g. 'moves e1'
  undo, u        revert the last move
  quit, q        exit

Move notation:
  e1-e2          deplacement (slide)
  e1>n           bascule (n/s/e/w)
  e1@l, e1@r     pivot (left/right)

Board glyphs:
  O pierre   + feuille   X ciseaux   ^ abri
  w… white piece   b… black piece   * = capitaine
  ## = empty dark square"
    );
}
