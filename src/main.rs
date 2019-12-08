#![allow(dead_code)]

use std::fs;
use std::collections::VecDeque;
use std::collections::HashMap;

#[derive(Eq, PartialEq, Clone, Copy)]
struct Point {
    x : i32,
    y : i32,
}

struct Line {
    p1 : Point,
    p2 : Point,
}

struct OrbitMap<'a> {
    map : HashMap<&'a str, &'a str>,
}

impl OrbitMap<'_> {
    fn new(input : &str) -> OrbitMap {
        let mut orbit_map = HashMap::new();

        for orbit in input.split('\n') {
            let pair : Vec<&str> = orbit.split(')').collect();
            orbit_map.insert(pair[1], pair[0]);
        }

        OrbitMap { map : orbit_map }
    }

    fn count_orbits(&self, orbiter : &str) -> usize {
        if orbiter == "COM" {
            0
        } else {
            self.count_orbits(self.map.get(&orbiter).unwrap()) + 1
        }
    }

    fn count_total_orbits(&self) -> usize {
        self.map.keys()
                .map(|o| self.count_orbits(o))
                .sum()
    }

    fn is_orbiting(&self, orbiter : &str, dst : &str) -> bool {
        if orbiter == "COM" {
            dst == "COM"
        } else if orbiter == dst {
            true
        } else {
            self.is_orbiting(self.map.get(&orbiter).unwrap(), dst)
        }
    }

    fn count_transfers(&self, from : &str, to : &str) -> usize {
        if from == to {
            0
        } else if self.is_orbiting(to, self.map.get(from).unwrap()) {
            self.count_transfers(to, self.map.get(from).unwrap()) + 1
        } else {
            self.count_transfers(self.map.get(from).unwrap(), to) + 1
        }
    }
}

impl Point {
    fn distance_origin(self) -> i32 {
        self.x.abs() + self.y.abs()
    }
}

impl Line {
    fn horizontal(&self) -> bool {
        self.p1.y == self.p2.y
    }

    fn length(&self) -> i32 {
        if self.horizontal() {
            (self.p1.x - self.p2.x).abs()
        } else {
            (self.p1.y - self.p2.y).abs()
        }
    }

    fn contains_point(&self, point : Point) -> bool {
        if self.horizontal() {
            if self.p1.y != point.y {
                return false;
            }
            self.p1.x <= point.x && self.p2.x >= point.x
            || self.p2.x <= point.x && self.p1.x >= point.x
        } else {
            if self.p1.x != point.x {
                return false;
            }
            self.p1.y <= point.y && self.p2.y >= point.y
            || self.p2.y <= point.y && self.p1.y >= point.y
        }
    }

    fn intersection(&self, other : &Line) -> Option<Point> {
        if self.horizontal() == other.horizontal() {
            return None;
        }

        let point = if self.horizontal() {
            Point { x : other.p1.x, y : self.p1.y }
        } else {
            Point { x : self.p1.x, y : other.p1.y }
        };

        if self.contains_point(point) && other.contains_point(point) {
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
                         .map(required_fuel)
                         .sum();
    println!("1a: {}", sum);

    let sum2 : u32 = input.split('\n')
                          .filter(|x| !x.is_empty())
                          .map(|x| x.parse::<u32>().unwrap())
                          .map(required_fuel_with_fuel)
                          .sum();
    println!("1b: {}", sum2);
}

#[derive(PartialEq)]
enum ProgramState {
    RUNNING,
    HALTED,
    SUSPENDED,
}

struct IntComputer {
    ip : usize,
    mem : Vec<i32>,
    input : VecDeque<i32>,
    output : VecDeque<i32>,
    suspend_on_output : bool,
}

impl IntComputer {
    fn new(program : &[i32]) -> IntComputer {
        IntComputer {
            ip : 0,
            mem : program.to_vec(),
            input : VecDeque::new(),
            output : VecDeque::new(),
            suspend_on_output : false,
        }
    }

    fn get_operands(&self, inputs : &[i32], mode : i32) -> Vec<i32> {
        let mut tmp = mode;
        let mut ops = Vec::new();
        for i in inputs {
            let m = tmp % 10;
            tmp /= 10;
            match m {
                0 => ops.push(self.mem[*i as usize]),
                1 => ops.push(*i),
                _ => panic!("invalid mode")
            }
        }
        ops
    }

    fn run_program(&mut self) -> ProgramState {
        loop {
            let state = self.single_step();
            if state != ProgramState::RUNNING {
                return state;
            }
        }
    }

    fn single_step(&mut self) -> ProgramState {
        let instruction = self.mem[self.ip];
        let opcode = instruction % 100;
        let mode = instruction / 100;

        match opcode {
            1 => { /* add */
                let ops = self.get_operands(&self.mem[self.ip+1..=self.ip+2], mode);
                let dst = self.mem[self.ip+3] as usize;
                self.mem[dst] = ops[0] + ops[1];
                self.ip += 4;
            }
            2 => { /* multiply */
                let ops = self.get_operands(&self.mem[self.ip+1..=self.ip+2], mode);
                let dst = self.mem[self.ip+3] as usize;
                self.mem[dst] = ops[0] * ops[1];
                self.ip += 4;
            }
            3 => { /* read stdin */
                let dst = self.mem[self.ip+1] as usize;
                self.mem[dst] = self.input.pop_front().expect("not enough input available");
                self.ip += 2;
            }
            4 => { /* write stdout */
                let ops = self.get_operands(&[self.mem[self.ip+1]], mode);
                self.output.push_back(ops[0]);
                self.ip += 2;
                if self.suspend_on_output {
                    return ProgramState::SUSPENDED;
                }
            }
            5 => { /* jump-if-true */
                let ops = self.get_operands(&self.mem[self.ip+1..=self.ip+2], mode);
                if ops[0] != 0 {
                    self.ip = ops[1] as usize;
                } else {
                    self.ip += 3;
                }
            }
            6 => { /* jump-if-false */
                let ops = self.get_operands(&self.mem[self.ip+1..=self.ip+2], mode);
                if ops[0] == 0 {
                    self.ip = ops[1] as usize;
                } else {
                    self.ip += 3;
                }
            }
            7 => { /* less-than */
                let ops = self.get_operands(&self.mem[self.ip+1..=self.ip+2], mode);
                let dst = self.mem[self.ip+3] as usize;
                if ops[0] < ops[1] {
                    self.mem[dst] = 1;
                } else {
                    self.mem[dst] = 0;
                }
                self.ip += 4;
            }
            8 => { /* equals */
                let ops = self.get_operands(&self.mem[self.ip+1..=self.ip+2], mode);
                let dst = self.mem[self.ip+3] as usize;
                if ops[0] == ops[1] {
                    self.mem[dst] = 1;
                } else {
                    self.mem[dst] = 0;
                }
                self.ip += 4;
            }
            99 => return ProgramState::HALTED,
            _ => panic!("invalid opcode")
        }
        ProgramState::RUNNING
    }
}


fn day2() {
    let input = read_file("input2");
    let input = input.trim_end();
    let mut program : Vec<i32> = input.split(',')
                                      .map(|x| x.parse::<i32>().unwrap())
                                      .collect();

    program[1] = 12;
    program[2] = 2;

    let mut computer = IntComputer::new(&program);
    computer.run_program();
    println!("2a: {}", computer.mem[0]);

    for noun in 0..=99 {
        for verb in 0..=99 {
            program[1] = noun;
            program[2] = verb;
            let mut computer = IntComputer::new(&program);
            computer.run_program();

            if computer.mem[0] == 19_690_720 {
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

fn intersections(lines1 : &[Line], lines2 : &[Line]) -> Vec<Point> {
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
                                   .filter(|x| x.is_positive())
                                   .min().unwrap()
}

fn delay_to_point(lines : &[Line], p : Point) -> i32 {
    let mut delay = 0;
    for line in lines {
        if line.contains_point(p) {
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
          .map(|p| delay_to_point(&lines1, *p) + delay_to_point(&lines2, *p))
          .filter(|d| d.is_positive())
          .min().unwrap()
}

fn day3() {
    let input = read_file("input3");
    let input = input.trim_end();
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
    let min = 264_360;
    let max = 746_325;

    let count = (min..=max).filter(|&pw| valid_password(pw)).count();
    println!("4a: {}", count);

    let count = (min..=max).filter(|&pw| valid_password2(pw)).count();
    println!("4a: {}", count);
}

fn day5() {
    let input = read_file("input5");
    let input = input.trim_end();
    let program : Vec<i32> = input.split(',')
                                  .map(|x| x.parse::<i32>().unwrap())
                                  .collect();
    let mut computer = IntComputer::new(&program);
    computer.input.push_back(1);
    computer.run_program();
    println!("5a:");
    for val in computer.output {
        println!("{}", val);
    }

    let mut computer = IntComputer::new(&program);
    computer.input.push_back(5);
    computer.run_program();
    println!("5b:");
    for val in computer.output {
        println!("{}", val);
    }
}

fn day6() {
    let input = read_file("input6");
    let input = input.trim_end();
    let orbit_map = OrbitMap::new(input);

    println!("6a: {}", orbit_map.count_total_orbits());

    println!("6b: {}", orbit_map.count_transfers(orbit_map.map["YOU"], orbit_map.map["SAN"]));
}

fn permutations(elements : Vec<i32>) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    if elements.len() == 1 {
        result.push(elements);
        return result;
    }
    for current_el in &elements {
        let mut tmp = elements.clone();
        let pos = tmp.iter().position(|x| x == current_el).unwrap();
        tmp.remove(pos);
        for mut p in permutations(tmp) {
            p.insert(0, *current_el);
            result.push(p);
        }
    }
    result
}

fn max_thruster_signal(program : &[i32]) -> i32 {
    let phase_settings = permutations(vec![0, 1, 2, 3, 4]);
    let mut signals = Vec::new();

    for setting in phase_settings {
        let mut output = 0;
        for phase in setting {
            let input = output;
            let mut computer = IntComputer::new(program);
            computer.input.push_back(phase);
            computer.input.push_back(input);
            computer.run_program();
            output = computer.output[0];
        }
        signals.push(output)
    }
    *signals.iter().max().unwrap()
}

fn max_thruster_signal_feedback(program : &[i32]) -> i32 {
    let phase_settings = permutations(vec![5, 6, 7, 8, 9]);
    let mut signals = Vec::new();

    for setting in phase_settings {
        let mut computers = Vec::new();
        for phase in setting {
            let mut computer = IntComputer::new(program);
            computer.suspend_on_output = true;
            computer.input.push_back(phase);
            computers.push(computer);
        }
        let mut output = 0;
        for i in (0..5).cycle() {
            let input = output;
            computers[i].input.push_back(input);
            match computers[i].run_program() {
                ProgramState::SUSPENDED => {
                    output = computers[i].output.pop_front().expect("not sufficient output available");
                }
                ProgramState::HALTED => {
                    if i == 4 {
                        /* last computer halted */
                        break;
                    }
                }
                _ => panic!("unexpected program state"),
            };
        }
        signals.push(output)
    }
    *signals.iter().max().unwrap()
}

fn day7() {
    let input = read_file("input7");
    let input = input.trim_end();
    let program : Vec<i32> = input.split(',')
                                  .map(|x| x.parse::<i32>().unwrap())
                                  .collect();

    println!("7a: {}", max_thruster_signal(&program));

    println!("7b: {}", max_thruster_signal_feedback(&program));
}

fn day8() {
    let input = read_file("input8");
    let input = input.trim_end();
    let pixels : Vec<u32> = input.chars()
                                 .map(|x| x.to_digit(10).unwrap())
                                 .collect();

    const WIDTH : usize = 25;
    const HEIGHT : usize = 6;

    let mut lowest = (i32::max_value(), 0);
    let mut image = [2; WIDTH * HEIGHT];
    for layer in pixels.chunks(WIDTH * HEIGHT) {
        let mut digits = HashMap::new();
        for i in 0..layer.len() {
            let pixel = layer[i];
            let count = digits.entry(pixel).or_insert(0);
            *count += 1;
            if image[i] == 2 {
                image[i] = pixel;
            }
        }
        let count0 = *digits.entry(0).or_insert(0);
        let count1 = *digits.entry(1).or_insert(0);
        let count2 = *digits.entry(2).or_insert(0);
        if count0 < lowest.0 {
            lowest.0 = count0;
            lowest.1 = count1 * count2;
        }
    }

    println!("8a: {}", lowest.1);

    println!("8b:");
    for row in image.chunks(WIDTH) {
        for pixel in row {
            match pixel {
                0 => print!("#"),
                1 => print!(" "),
                _ => panic!("invalid pixel"),
            };
        }
        println!();
    }
}

fn main() {
    day8();
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
        let mut computer = IntComputer::new(&[1,0,0,0,99]);
        computer.run_program();
        assert_eq!(computer.mem, vec!(2,0,0,0,99));

        let mut computer = IntComputer::new(&[2,4,4,5,99,0]);
        computer.run_program();
        assert_eq!(computer.mem, vec!(2,4,4,5,99,9801));

        let mut computer = IntComputer::new(&[1,1,1,4,99,5,6,0,99]);
        computer.run_program();
        assert_eq!(computer.mem, vec!(30,1,1,4,2,5,6,0,99));
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

        let mut computer = IntComputer::new(&program);
        computer.input.push_back(4);
        computer.run_program();
        assert_eq!(computer.output[0], 999);

        let mut computer = IntComputer::new(&program);
        computer.input.push_back(8);
        computer.run_program();
        assert_eq!(computer.output[0], 1000);

        let mut computer = IntComputer::new(&program);
        computer.input.push_back(42);
        computer.run_program();
        assert_eq!(computer.output[0], 1001);
    }

    #[test]
    fn test_day6() {
        let input = "COM)B\nB)C\nC)D\nD)E\nE)F\nB)G\nG)H\nD)I\nE)J\nJ)K\nK)L";
        let orbit_map = OrbitMap::new(input);
        assert_eq!(orbit_map.count_total_orbits(), 42);

        let input = input.to_string() + "\nK)YOU\nI)SAN";
        let orbit_map = OrbitMap::new(&input);
        assert_eq!(orbit_map.count_transfers(orbit_map.map["YOU"], orbit_map.map["SAN"]), 4);
    }

    #[test]
    fn test_day7() {
        let program = vec![3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0];
        assert_eq!(max_thruster_signal(&program), 43210);

        let program = vec![3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,23,99,0,0];
        assert_eq!(max_thruster_signal(&program), 54321);

        let program = vec![3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0];
        assert_eq!(max_thruster_signal(&program), 65210);

        let program = vec![3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5];
        assert_eq!(max_thruster_signal_feedback(&program), 139629729);

        let program = vec![3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,-5,54,1105,1,12,1,53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10];
        assert_eq!(max_thruster_signal_feedback(&program), 18216);
    }
}
