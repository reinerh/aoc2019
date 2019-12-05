#![allow(dead_code)]

use std::fs;

#[derive(Eq, PartialEq, Clone, Copy)]
struct Point {
    x : i32,
    y : i32,
}

struct Line {
    p1 : Point,
    p2 : Point,
}

impl Point {
    fn distance_origin(&self) -> i32 {
        self.x.abs() + self.y.abs()
    }
}

impl Line {
    fn horizontal(&self) -> bool {
        self.p1.y == self.p2.y
    }

    fn length(&self) -> i32 {
        match self.horizontal() {
            true => (self.p1.x - self.p2.x).abs(),
            false => (self.p1.y - self.p2.y).abs(),
        }
    }

    fn contains_point(&self, point : &Point) -> bool {
        if self.horizontal() && self.p1.y != point.y {
            return false;
        } else if !self.horizontal() && self.p1.x != point.x {
            return false;
        }

        match self.horizontal() {
            true => {
                self.p1.x <= point.x && self.p2.x >= point.x
             || self.p2.x <= point.x && self.p1.x >= point.x
            }
            false => {
                self.p1.y <= point.y && self.p2.y >= point.y
             || self.p2.y <= point.y && self.p1.y >= point.y
            }
        }
    }

    fn intersection(&self, other : &Line) -> Option<Point> {
        if self.horizontal() == other.horizontal() {
            return None;
        }

        let point = match self.horizontal() {
            true => Point { x : other.p1.x, y : self.p1.y },
            false => Point { x : self.p1.x, y : other.p1.y },
        };

        if self.contains_point(&point) && other.contains_point(&point) {
            Some(point)
        } else {
            None
        }
    }
}

fn read_file(file : &str) -> String {
    fs::read_to_string(file).unwrap()
}

fn required_fuel(mass : u32) -> u32 {
    (mass / 3).saturating_sub(2)
}

fn required_fuel_with_fuel(mass : u32) -> u32 {
    let mut total = 0;
    let mut current = mass;
    loop {
        current = required_fuel(current);
        total += current;
        if current == 0 {
            break;
        }
    }
    total
}

fn day1() {
    let input = read_file("input1");
    let sum : u32 = input.split('\n')
                      .filter(|x| !x.is_empty())
                      .map(|x| x.parse::<u32>().unwrap())
                      .map(|x| required_fuel(x))
                      .sum();
    println!("1a: {}", sum);

    let sum2 : u32 = input.split('\n')
                         .filter(|x| !x.is_empty())
                         .map(|x| x.parse::<u32>().unwrap())
                         .map(|x| required_fuel_with_fuel(x))
                         .sum();
    println!("1b: {}", sum2);
}

fn get_operands(mem : &Vec<i32>, inputs : Vec<i32>, mode : i32) -> Vec<i32> {
    let mut tmp = mode;
    let mut ops = Vec::new();
    for i in inputs {
        let m = tmp % 10;
        tmp /= 10;
        match m {
            0 => ops.push(mem[i as usize]),
            1 => ops.push(i),
            _ => panic!("invalid mode")
        }
    }
    ops
}

fn run_program_io(input : Vec<i32>, stdin : &Vec<&str>, stdout : &mut Vec<i32>) -> Vec<i32> {
    let mut stdin_iter = stdin.iter();
    let mut mem = input.clone();
    let mut ip = 0;
    loop {
        let instruction = mem[ip];
        let opcode = instruction % 100;
        let mode = instruction / 100;

        match opcode {
            1 => { /* add */
                let ops = get_operands(&mem, mem[ip+1..ip+3].to_vec(), mode);
                let dst = mem[ip+3] as usize;
                mem[dst] = ops[0] + ops[1];
                ip += 4;
            }
            2 => { /* multiply */
                let ops = get_operands(&mem, mem[ip+1..ip+3].to_vec(), mode);
                let dst = mem[ip+3] as usize;
                mem[dst] = ops[0] * ops[1];
                ip += 4;
            }
            3 => { /* read stdin */
                let input = stdin_iter.next().unwrap();
                let dst = mem[ip+1] as usize;
                mem[dst] = input.parse::<i32>().unwrap();
                ip += 2;
            }
            4 => { /* write stdout */
                let ops = get_operands(&mem, mem[ip+1..ip+2].to_vec(), mode);
                stdout.push(ops[0]);
                ip += 2;
            }
            5 => { /* jump-if-true */
                let ops = get_operands(&mem, mem[ip+1..ip+3].to_vec(), mode);
                if ops[0] != 0 {
                    ip = ops[1] as usize;
                } else {
                    ip += 3;
                }
            }
            6 => { /* jump-if-false */
                let ops = get_operands(&mem, mem[ip+1..ip+3].to_vec(), mode);
                if ops[0] == 0 {
                    ip = ops[1] as usize;
                } else {
                    ip += 3;
                }
            }
            7 => { /* less-than */
                let ops = get_operands(&mem, mem[ip+1..ip+3].to_vec(), mode);
                let dst = mem[ip+3] as usize;
                if ops[0] < ops[1] {
                    mem[dst] = 1;
                } else {
                    mem[dst] = 0;
                }
                ip += 4;
            }
            8 => { /* equals */
                let ops = get_operands(&mem, mem[ip+1..ip+3].to_vec(), mode);
                let dst = mem[ip+3] as usize;
                if ops[0] == ops[1] {
                    mem[dst] = 1;
                } else {
                    mem[dst] = 0;
                }
                ip += 4;
            }
            99 => break,
            _ => panic!("invalid opcode")
        }
    }
    mem
}

fn run_program(input : Vec<i32>) -> Vec<i32> {
    let stdin = Vec::new();
    let mut stdout = Vec::new();
    run_program_io(input, &stdin, &mut stdout)
}

fn day2() {
    let mut input = read_file("input2");
    input.pop();
    let mut program : Vec<i32> = input.split(',')
                                      .map(|x| x.parse::<i32>().unwrap())
                                      .collect();

    program[1] = 12;
    program[2] = 2;

    let mem = run_program(program.clone());
    println!("2a: {}", mem[0]);

    for noun in 0..99 {
        for verb in 0..99 {
            program[1] = noun;
            program[2] = verb;
            let mem = run_program(program.clone());

            if mem[0] == 19690720 {
                println!("2b: {}", 100 * noun + verb);
                return;
            }
        }
    }
}

fn parse_lines(wire : &str) -> Vec<Line> {
    let mut lines = Vec::new();

    let mut p1 = Point { x : 0, y : 0 };
    for next in wire.split(',') {
        let dist = next[1..].parse::<i32>().unwrap();

        let mut p2 = Point { x : p1.x, y : p1.y };
        match next.chars().next().unwrap() {
            'R' => p2.x += dist,
            'L' => p2.x -= dist,
            'U' => p2.y += dist,
            'D' => p2.y -= dist,
            _ => panic!("unexpected direction"),
        };
        lines.push(Line { p1, p2 });
        p1 = p2;
    }

    lines
}

fn intersections(lines1 : &Vec<Line>, lines2 : &Vec<Line>) -> Vec<Point> {
    let mut points = Vec::new();
    for line1 in lines1 {
        for line2 in lines2 {
            if let Some(p) = line1.intersection(line2) {
                points.push(p);
            }
        }
    }

    points
}

fn closest_intersection(wire1 : &str, wire2 : &str) -> i32 {
    let lines1 = parse_lines(wire1);
    let lines2 = parse_lines(wire2);
    intersections(&lines1, &lines2).iter()
                                   .map(|p| p.distance_origin())
                                   .filter(|&x| x > 0)
                                   .min().unwrap()
}

fn delay_to_point(lines : &Vec<Line>, p : &Point) -> i32 {
    let mut delay = 0;
    for line in lines {
        if line.contains_point(&p) {
            delay += (line.p1.distance_origin() - p.distance_origin()).abs();
            break;
        }
        delay += line.length();
    }

    delay
}

fn min_delay(wire1 : &str, wire2 : &str) -> i32 {
    let lines1 = parse_lines(wire1);
    let lines2 = parse_lines(wire2);
    let points = intersections(&lines1, &lines2);

    points.iter()
          .map(|p| delay_to_point(&lines1, &p) + delay_to_point(&lines2, &p))
          .filter(|&d| d > 0)
          .min().unwrap()
}

fn day3() {
    let mut input = read_file("input3");
    input.pop();
    let wires : Vec<&str> = input.split('\n').collect();

    let distance = closest_intersection(wires[0], wires[1]);
    println!("3a: {}", distance);

    let delay = min_delay(wires[0], wires[1]);
    println!("3b: {}", delay);
}

fn digits(input : u32) -> Vec<u32> {
    let mut tmp = input;
    let mut digits = Vec::new();
    loop {
        digits.push(tmp % 10);
        tmp /= 10;
        if tmp == 0 {
            break;
        }
    }
    digits.reverse();
    digits
}

fn valid_password(password : u32) -> bool {
    let digits = digits(password);
    if digits.len() != 6 {
        return false;
    }
    let mut prev = 0;
    let mut has_double = false;
    for d in digits {
        if d < prev {
            return false;
        }
        if d == prev {
            has_double = true;
        }
        prev = d
    }

    has_double
}

fn valid_password2(input : u32) -> bool {
    if !valid_password(input) {
        return false;
    }

    let digits = digits(input);
    let mut same_count = 1;
    let mut prev = 0;
    for d in digits {
        if d == prev {
            same_count += 1;
        } else {
            if same_count == 2 {
                return true;
            }
            same_count = 1;
        }
        prev = d;
    }
    same_count == 2
}

fn day4() {
    let min = 264360;
    let max = 746325;

    let count = (min..max+1).filter(|&pw| valid_password(pw)).count();
    println!("4a: {}", count);

    let count = (min..max+1).filter(|&pw| valid_password2(pw)).count();
    println!("4a: {}", count);
}

fn day5() {
    let mut input = read_file("input5");
    input.pop();
    let program : Vec<i32> = input.split(',')
                                  .map(|x| x.parse::<i32>().unwrap())
                                  .collect();
    let mut stdout = Vec::new();
    let stdin = vec!["1"];
    run_program_io(program.clone(), &stdin, &mut stdout);
    println!("5a:");
    for val in stdout {
        println!("{}", val);
    }

    let mut stdout = Vec::new();
    let stdin = vec!["5"];
    run_program_io(program.clone(), &stdin, &mut stdout);
    println!("5b:");
    for val in stdout {
        println!("{}", val);
    }
}

fn main() {
    day5();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_day1() {
        assert_eq!(required_fuel(12), 2);
        assert_eq!(required_fuel(14), 2);
        assert_eq!(required_fuel(1969), 654);
        assert_eq!(required_fuel(100756), 33583);

        assert_eq!(required_fuel_with_fuel(14), 2);
        assert_eq!(required_fuel_with_fuel(1969), 966);
        assert_eq!(required_fuel_with_fuel(100756), 50346);
    }

    #[test]
    fn test_day2() {
        assert_eq!(run_program(vec!(1,0,0,0,99)), vec!(2,0,0,0,99));
        assert_eq!(run_program(vec!(2,3,0,3,99)), vec!(2,3,0,6,99));
        assert_eq!(run_program(vec!(2,4,4,5,99,0)), vec!(2,4,4,5,99,9801));
        assert_eq!(run_program(vec!(1,1,1,4,99,5,6,0,99)), vec!(30,1,1,4,2,5,6,0,99));
    }

    #[test]
    fn test_day3() {
        let wire1 = "R8,U5,L5,D3";
        let wire2 = "U7,R6,D4,L4";
        assert_eq!(closest_intersection(wire1, wire2), 6);
        assert_eq!(min_delay(wire1, wire2), 30);

        let wire1 = "R75,D30,R83,U83,L12,D49,R71,U7,L72";
        let wire2 = "U62,R66,U55,R34,D71,R55,D58,R83";
        assert_eq!(closest_intersection(wire1, wire2), 159);
        assert_eq!(min_delay(wire1, wire2), 610);

        let wire1 = "R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51";
        let wire2 = "U98,R91,D20,R16,D67,R40,U7,R15,U6,R7";
        assert_eq!(closest_intersection(wire1, wire2), 135);
        assert_eq!(min_delay(wire1, wire2), 410);
    }

    #[test]
    fn test_day4() {
        assert!(valid_password(111111));
        assert!(!valid_password(223450));
        assert!(!valid_password(123789));

        assert!(valid_password2(112233));
        assert!(!valid_password2(123444));
        assert!(valid_password2(111122));
    }

    #[test]
    fn test_day5() {
        let program = vec![3,21,1008,21,8,20,1005,20,22,107,8,21,20,1006,20,31,1106,0,36,98,0,0,1002,21,125,20,4,20,1105,1,46,104,999,1105,1,46,1101,1000,1,20,4,20,1105,1,46,98,99];

        let stdin = vec!["4"];
        let mut stdout = Vec::new();
        run_program_io(program.clone(), &stdin, &mut stdout);
        assert_eq!(stdout[0], 999);

        let stdin = vec!["8"];
        let mut stdout = Vec::new();
        run_program_io(program.clone(), &stdin, &mut stdout);
        assert_eq!(stdout[0], 1000);

        let stdin = vec!["42"];
        let mut stdout = Vec::new();
        run_program_io(program.clone(), &stdin, &mut stdout);
        assert_eq!(stdout[0], 1001);
    }
}
