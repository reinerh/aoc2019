#![allow(dead_code)]

use std::fs::{self, File};
use std::io::prelude::*;
use std::cmp;
use std::cmp::Ordering;
use std::collections::{VecDeque, HashMap, HashSet};

#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash)]
struct Point {
    x : i32,
    y : i32,
}

#[derive(Copy, Clone, PartialEq)]
enum Direction {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3,
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

    fn distance(self, to: Point) -> Point {
        Point {
            x: to.x - self.x,
            y: to.y - self.y,
        }
    }

    fn move_in_direction(&mut self, direction: Direction) {
        match direction {
            Direction::UP => self.y += 1,
            Direction::RIGHT => self.x += 1,
            Direction::DOWN => self.y -= 1,
            Direction::LEFT => self.x -= 1,
        };
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
    NEEDINPUT,
}

struct Screen {
    screen: HashMap<(isize, isize), isize>,
    tiles_map_ascii: HashMap<isize, char>,
    tiles_map_color: HashMap<isize, (u8, u8, u8)>,
}

impl Screen {
    fn new() -> Screen {
        Screen {
            screen: HashMap::new(),
            tiles_map_ascii: HashMap::new(),
            tiles_map_color: HashMap::new(),
        }
    }

    fn set_pixel(&mut self, posx: isize, posy: isize, val: isize) {
        self.screen.insert((posx, posy), val);
    }

    fn dump_screen(&self, index: Option<isize>) {
        if self.screen.is_empty() {
            return;
        }
        let mut min_x = isize::max_value();
        let mut min_y = isize::max_value();
        let mut max_x = isize::min_value();
        let mut max_y = isize::min_value();
        for (x, y) in self.screen.keys() {
            min_x = cmp::min(min_x, *x);
            min_y = cmp::min(min_y, *y);
            max_x = cmp::max(max_x, *x);
            max_y = cmp::max(max_y, *y);
        }
        let mut screen_vec = Vec::new();
        for _ in 0..=(max_y - min_y) {
            let mut row = Vec::new();
            row.resize((max_x - min_x + 1) as usize, 0);
            screen_vec.push(row);
        }
        for (x, y) in self.screen.keys() {
            let mut tile = self.screen.get(&(*x,*y));
            if tile.is_none() {
                tile = Some(&0);
            }
            screen_vec[(y - min_y) as usize][(x - min_x) as usize] = *tile.unwrap();
        }
        if let Some(idx) = index {
            /* write to file */
            let mut file = File::create(format!("screen{:05}.ppm", idx)).expect("can't open file for writing");
            file.write_all(format!("P6\n{} {} 255\n", screen_vec[0].len(), screen_vec.len()).as_bytes()).unwrap();
            for row in screen_vec {
                for tile in row {
                    let (r, g, b) = self.tiles_map_color.get(&tile).unwrap();
                    file.write_all(&[*r, *g, *b]).unwrap();
                }
            }
        } else {
            /* dump to stdout */
            for row in screen_vec {
                for tile in row {
                    match self.tiles_map_ascii.get(&tile) {
                        Some(x) => print!("{}", x),
                        None => print!("{}", tile),
                    }
                }
                println!();
            }
        }
    }
}

struct IntComputer {
    ip : usize,
    mem : Vec<isize>,
    base_address : isize,
    input : VecDeque<isize>,
    output : VecDeque<isize>,
    suspend_on_output : bool,
    screen: Screen,
}

impl IntComputer {
    fn new(program : &[isize]) -> IntComputer {
        IntComputer {
            ip : 0,
            mem : program.to_vec(),
            base_address : 0,
            input : VecDeque::new(),
            output : VecDeque::new(),
            suspend_on_output : false,
            screen: Screen::new(),
        }
    }

    fn read_mem(&mut self, i : usize) -> isize {
        if i >= self.mem.len() {
            self.mem.resize(i+1, 0);
        }
        self.mem[i]
    }

    fn write_mem(&mut self, i : isize, val : isize, mode : isize) {
        let pos = match mode {
            0 => i,
            1 => panic!("writing not supported in immediate mode"),
            2 => self.base_address + i,
            _ => panic!("unimplemented mode")
        } as usize;
        if pos >= self.mem.len() {
            self.mem.resize(pos+1, 0);
        }
        self.mem[pos] = val;
    }

    fn get_operands(&mut self, count : usize, mode : isize) -> Vec<isize> {
        let inputs = self.mem[self.ip+1..=self.ip+count].to_vec();
        let mut tmp = mode;
        let mut ops = Vec::new();
        for i in inputs {
            let m = tmp % 10;
            tmp /= 10;
            match m {
                0 => ops.push(self.read_mem(i as usize)),
                1 => ops.push(i),
                2 => ops.push(self.read_mem((self.base_address + i) as usize)),
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
                let ops = self.get_operands(2, mode);
                let dst = self.mem[self.ip+3] as isize;
                self.write_mem(dst, ops[0] + ops[1], mode/100);
                self.ip += 4;
            }
            2 => { /* multiply */
                let ops = self.get_operands(2, mode);
                let dst = self.mem[self.ip+3] as isize;
                self.write_mem(dst, ops[0] * ops[1], mode/100);
                self.ip += 4;
            }
            3 => { /* read stdin */
                if self.input.is_empty() {
                    return ProgramState::NEEDINPUT;
                }
                let dst = self.mem[self.ip+1] as isize;
                let input = self.input.pop_front().expect("not enough input available");
                self.write_mem(dst, input, mode);
                self.ip += 2;
            }
            4 => { /* write stdout */
                let ops = self.get_operands(1, mode);
                self.output.push_back(ops[0]);
                self.ip += 2;
                if self.suspend_on_output {
                    return ProgramState::SUSPENDED;
                }
            }
            5 => { /* jump-if-true */
                let ops = self.get_operands(2, mode);
                if ops[0] != 0 {
                    self.ip = ops[1] as usize;
                } else {
                    self.ip += 3;
                }
            }
            6 => { /* jump-if-false */
                let ops = self.get_operands(2, mode);
                if ops[0] == 0 {
                    self.ip = ops[1] as usize;
                } else {
                    self.ip += 3;
                }
            }
            7 => { /* less-than */
                let ops = self.get_operands(2, mode);
                let dst = self.mem[self.ip+3] as isize;
                if ops[0] < ops[1] {
                    self.write_mem(dst, 1, mode/100);
                } else {
                    self.write_mem(dst, 0, mode/100);
                }
                self.ip += 4;
            }
            8 => { /* equals */
                let ops = self.get_operands(2, mode);
                let dst = self.mem[self.ip+3] as isize;
                if ops[0] == ops[1] {
                    self.write_mem(dst, 1, mode/100);
                } else {
                    self.write_mem(dst, 0, mode/100);
                }
                self.ip += 4;
            }
            9 => { /* adjust relative base */
                let ops = self.get_operands(1, mode);
                self.base_address += ops[0];
                self.ip += 2;
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
    let mut program : Vec<isize> = input.split(',')
                                        .map(|x| x.parse::<isize>().unwrap())
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
    let program : Vec<isize> = input.split(',')
                                    .map(|x| x.parse::<isize>().unwrap())
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

fn permutations(elements : Vec<isize>) -> Vec<Vec<isize>> {
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

fn max_thruster_signal(program : &[isize]) -> isize {
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

fn max_thruster_signal_feedback(program : &[isize]) -> isize {
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
    let program : Vec<isize> = input.split(',')
                                    .map(|x| x.parse::<isize>().unwrap())
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

fn day9() {
    let input = read_file("input9");
    let input = input.trim_end();
    let program : Vec<isize> = input.split(',')
                                    .map(|x| x.parse::<isize>().unwrap())
                                    .collect();

    let mut computer = IntComputer::new(&program);
    computer.input.push_back(1);
    computer.run_program();
    println!("9a: {}", computer.output[0]);

    let mut computer = IntComputer::new(&program);
    computer.input.push_back(2);
    computer.run_program();
    println!("9b: {}", computer.output[0]);
}

struct AsteroidMap {
    asteroids: Vec<Point>,
}

impl AsteroidMap {
    fn new(input: &str) -> AsteroidMap {
        let mut asteroids = Vec::new();
        let rows: Vec<&str> = input.split('\n').collect();

        for (y, row) in rows.iter().enumerate() {
            for x in 0..row.len() {
                if row.as_bytes()[x] as char == '#' {
                    let pos = Point { x: x as i32, y: y as i32 };
                    asteroids.push(pos);
                }
            }
        }
        AsteroidMap { asteroids }
    }

    fn can_see(&self, from: Point, to: Point) -> bool {
        if from == to {
            /* ignore yourself */
            return false
        }
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        for div in 2..=cmp::min(dx.abs(), dy.abs()) {
            if (dx % div).abs() > 0 || (dy % div).abs() > 0 {
                /* only check divisible positions without remainder */
                continue;
            }
            let stepx = dx / div;
            let stepy = dy / div;
            let mut pos = Point {
                x: from.x + stepx,
                y: from.y + stepy,
            };
            while pos != to {
                if self.asteroids.contains(&pos) {
                    /* another asteroid found in line of sight */
                    return false;
                }
                pos.x += stepx;
                pos.y += stepy;
            }
        }
        if dx == 0 || dy == 0 {
            let line = Line { p1: from, p2: to };
            !self.asteroids.iter()
                           .filter(|&asteroid| *asteroid != from && *asteroid != to)
                           .any(|&asteroid| line.contains_point(asteroid))
        } else {
            true
        }
    }

    fn seen_asteroids(&self, pos: Point) -> Vec<Point> {
        self.asteroids.iter()
                      .filter(|&asteroid| self.can_see(pos, *asteroid))
                      .copied()
                      .collect()
    }

    fn max_detected_asteroids(&self) -> (Point, usize) {
        self.asteroids.iter()
                      .map(|&asteroid| (asteroid, self.seen_asteroids(asteroid).len()))
                      .max_by_key(|x| x.1).unwrap()
    }

    fn asteroid_angle(from: Point, to: Point) -> f32 {
        // point relative to 'from' as center
        // convert to "normal" coordinates with y axis from bottom->up
        let p = Point { x: to.x - from.x, y: -(to.y - from.y) };

        // calculate angle between (0, 1) and vector to new point
        let angle = f32::acos((p.y as f32) / f32::sqrt((p.x*p.x + p.y*p.y) as f32)).to_degrees();
        if from.x > to.x {
            360.0 - angle
        } else {
            angle
        }
    }

    fn asteroid_angle_cmp(from: Point, p1: Point, p2: Point) -> Ordering {
        if p1 == p2 {
            return Ordering::Equal;
        }
        let a1 = AsteroidMap::asteroid_angle(from, p1);
        let a2 = AsteroidMap::asteroid_angle(from, p2);
        a1.partial_cmp(&a2).unwrap()
    }

    fn vaporized(&mut self, from: Point, count: usize) -> Point {
        let mut removed = 0;
        loop {
            let mut seen = self.seen_asteroids(from);
            seen.sort_by(|&a1, &a2| AsteroidMap::asteroid_angle_cmp(from, a1, a2));
            for asteroid in &seen {
                let map_pos = self.asteroids.iter().position(|x| x == asteroid).unwrap();
                self.asteroids.remove(map_pos);
                removed += 1;
                if removed == count {
                    return *asteroid;
                }
            }
        }
    }
}

fn day10() {
    let input = read_file("input10");
    let input = input.trim_end();
    let mut asteroid_map = AsteroidMap::new(input);

    let max = asteroid_map.max_detected_asteroids();
    println!("10a: {}", max.1);

    let vaporized = asteroid_map.vaporized(max.0, 200);
    println!("10b: {}", vaporized.x * 100 + vaporized.y);
}

struct RobotDirection {
    direction: Direction,
}

impl RobotDirection {
    fn rotate(&mut self, direction: isize) {
        self.direction = match direction {
            0 => {
                match &self.direction {
                    Direction::UP => Direction::LEFT,
                    Direction::LEFT => Direction::DOWN,
                    Direction::DOWN => Direction::RIGHT,
                    Direction::RIGHT => Direction::UP,
                }
            }
            1 => {
                match &self.direction {
                    Direction::UP => Direction::RIGHT,
                    Direction::RIGHT => Direction::DOWN,
                    Direction::DOWN => Direction::LEFT,
                    Direction::LEFT => Direction::UP,
                }
            }
            _ => panic!("invalid direction")
        }
    }
}

fn hull_paint(program: &[isize], starting_color: isize) -> HashMap<Point, isize> {
    let mut computer = IntComputer::new(program);
    computer.suspend_on_output = true;

    let mut hull = HashMap::new();
    let mut pos = Point { x: 0, y: 0 };
    let mut direction = RobotDirection { direction: Direction::UP };
    let mut new_color = None;
    let mut rotation = None;
    hull.insert(pos, starting_color);
    loop {
        match computer.run_program() {
            ProgramState::SUSPENDED => {
                let output = computer.output.pop_front().unwrap();
                if new_color.is_none() {
                    new_color = Some(output);
                } else if rotation.is_none() {
                    rotation = Some(output);
                } else {
                    panic!("unexpected state");
                }
            }
            ProgramState::NEEDINPUT => {
                computer.input.push_back(*hull.entry(pos).or_insert(0));
                continue;
            }
            ProgramState::HALTED => break,
            _ => {}
        }
        if new_color.is_some() && rotation.is_some() {
            hull.insert(pos, new_color.unwrap());
            direction.rotate(rotation.unwrap());
            pos.move_in_direction(direction.direction);
            new_color = None;
            rotation = None;
        }
    }
    hull
}

fn day11() {
    let input = read_file("input11");
    let input = input.trim_end();
    let program : Vec<isize> = input.split(',')
                                    .map(|x| x.parse::<isize>().unwrap())
                                    .collect();

    let hull = hull_paint(&program, 0);
    println!("11a: {}", hull.len());

    let hull = hull_paint(&program, 1);
    let mut min_x = i32::max_value();
    let mut min_y = i32::max_value();
    let mut max_x = i32::min_value();
    let mut max_y = i32::min_value();
    for pos in hull.keys() {
        min_x = cmp::min(min_x, pos.x);
        min_y = cmp::min(min_y, pos.y);
        max_x = cmp::max(max_x, pos.x);
        max_y = cmp::max(max_y, pos.y);
    }
    let mut hull_vec = Vec::new();
    for _ in 0..=(max_y - min_y) {
        let mut row = Vec::new();
        row.resize((max_x - min_x + 1) as usize, 0);
        hull_vec.push(row);
    }
    for (pos, color) in &hull {
        hull_vec[(pos.y - min_y) as usize][(pos.x - min_x) as usize] = *color;
    }
    hull_vec.reverse();

    println!("11b:");
    for row in hull_vec {
        for color in row {
            match color {
                0 => print!("."),
                1 => print!("#"),
                _ => panic!("invalid color")
            }
        }
        println!();
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
struct Position {
    x: isize,
    y: isize,
    z: isize,
}

#[derive(PartialEq, Eq, Clone, Copy)]
struct Moon {
    pos: Position,
    velocity: Position,
}

impl Moon {
    fn new(x: isize, y: isize, z: isize) -> Moon {
        Moon {
            pos: Position { x, y, z },
            velocity: Position { x: 0, y: 0, z: 0 },
        }
    }

    fn apply_gravity(&mut self, other: &Moon) {
        self.velocity.x += (other.pos.x - self.pos.x).signum();
        self.velocity.y += (other.pos.y - self.pos.y).signum();
        self.velocity.z += (other.pos.z - self.pos.z).signum();
    }

    fn update_position(&mut self) {
        self.pos.x += self.velocity.x;
        self.pos.y += self.velocity.y;
        self.pos.z += self.velocity.z;
    }

    fn potential_energy(&self) -> isize {
        self.pos.x.abs() + self.pos.y.abs() + self.pos.z.abs()
    }

    fn kinetic_energy(&self) -> isize {
        self.velocity.x.abs() + self.velocity.y.abs() + self.velocity.z.abs()
    }

    fn total_energy(&self) -> isize {
        self.potential_energy() * self.kinetic_energy()
    }
}

fn simulate_moons(moons: &mut Vec<Moon>) {
    for i in 0..moons.len() {
        for j in 0..moons.len() {
            let other = moons[j];
            moons[i].apply_gravity(&other);
        }
    }
    for moon in moons {
        moon.update_position();
    }
}

fn gcd(a: isize, b: isize) -> isize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: isize, b: isize) -> isize {
    (a*b).abs() / gcd(a, b)
}

fn day12() {
    let initial_state = vec![
        Moon::new(17, -9,   4),
        Moon::new( 2,  2, -13),
        Moon::new(-1,  5,  -1),
        Moon::new( 4,  7,  -7),
    ];
    let mut moons = initial_state;

    for _ in 0..1000 {
        simulate_moons(&mut moons);
    }

    let total_energy: isize = moons.iter()
                                   .map(|moon| moon.total_energy())
                                   .sum();
    println!("12a: {}", total_energy);

    let mut period_x = 0;
    let initial_x_values: Vec<(isize, isize)> = moons.iter().map(|moon| (moon.pos.x, moon.velocity.x)).collect();
    loop {
        period_x += 1;
        simulate_moons(&mut moons);
        let x_values: Vec<(isize, isize)> = moons.iter().map(|moon| (moon.pos.x, moon.velocity.x)).collect();
        if x_values == initial_x_values {
            break;
        }
    }

    let mut period_y = 0;
    let initial_y_values: Vec<(isize, isize)> = moons.iter().map(|moon| (moon.pos.y, moon.velocity.y)).collect();
    loop {
        period_y += 1;
        simulate_moons(&mut moons);
        let y_values: Vec<(isize, isize)> = moons.iter().map(|moon| (moon.pos.y, moon.velocity.y)).collect();
        if y_values == initial_y_values {
            break;
        }
    }

    let mut period_z = 0;
    let initial_z_values: Vec<(isize, isize)> = moons.iter().map(|moon| (moon.pos.z, moon.velocity.z)).collect();
    loop {
        period_z += 1;
        simulate_moons(&mut moons);
        let z_values: Vec<(isize, isize)> = moons.iter().map(|moon| (moon.pos.z, moon.velocity.z)).collect();
        if z_values == initial_z_values {
            break;
        }
    }

    println!("12b: {}", lcm(lcm(period_x, period_y), period_z));
}

struct Arcade {
    computer: IntComputer,
    score: isize,
}

impl Arcade {
    fn new(program: &[isize]) -> Arcade {
        let mut tiles_map = HashMap::new();
        tiles_map.insert(0, (255, 255, 255));  /* empty */
        tiles_map.insert(1, ( 32,  32,  32));  /* wall */
        tiles_map.insert(2, (127, 127, 127));  /* block */
        tiles_map.insert(3, (255,   0,   0));  /* paddle */
        tiles_map.insert(4, (  0,   0, 255));  /* ball */

        let mut computer = IntComputer::new(&program);
        computer.screen.tiles_map_color = tiles_map;

        Arcade {
            computer,
            score: 0,
        }
    }

    fn find_tile(&self, search: isize) -> (isize, isize) {
        for (pos, tile) in &self.computer.screen.screen {
            if *tile == search {
                return *pos;
            }
        }
        panic!("tile not found on screen");
    }

    fn find_ball(&self) -> (isize, isize) {
        self.find_tile(4)
    }

    fn find_paddle(&self) -> (isize, isize) {
        self.find_tile(3)
    }

    fn run_arcade(&mut self) {
        let mut x = None;
        let mut y = None;
        let mut tile = None;
        loop {
            match self.computer.run_program() {
                ProgramState::SUSPENDED => {
                    let out = self.computer.output.pop_front().unwrap();
                    if x.is_none() {
                        x = Some(out);
                    } else if y.is_none() {
                        y = Some(out);
                    } else if tile.is_none() {
                        tile = Some(out);
                    } else {
                        panic!("invalid state");
                    }
                }
                ProgramState::NEEDINPUT => {
                    let pos_ball = self.find_ball();
                    let pos_paddle = self.find_paddle();
                    let input = (pos_ball.0 - pos_paddle.0).signum();
                    self.computer.input.push_back(input);
                }
                ProgramState::HALTED => break,
                _ => {}
            };
            if let (Some(posx), Some(posy), Some(new_tile)) = (x, y, tile) {
                if posx == -1 && posy == 0 {
                    self.score = new_tile;
                } else {
                    self.computer.screen.set_pixel(posx, posy, new_tile);
                }
                x = None;
                y = None;
                tile = None;
            }
        }
    }
}

fn day13() {
    let input = read_file("input13");
    let input = input.trim_end();
    let program : Vec<isize> = input.split(',')
                                    .map(|x| x.parse::<isize>().unwrap())
                                    .collect();

    let mut arcade = Arcade::new(&program);
    arcade.computer.suspend_on_output = true;
    arcade.run_arcade();
    let blocks = arcade.computer.screen.screen.values()
                                              .filter(|&tile| *tile == 2)
                                              .count();

    println!("13a: {}", blocks);

    let mut arcade = Arcade::new(&program);
    arcade.computer.suspend_on_output = true;
    arcade.computer.mem[0] = 2;
    arcade.run_arcade();
    println!("13b: {}", arcade.score);
}

struct Reaction {
    result: String,
    amount: isize,
    resources: HashMap<String, isize>,
}

impl Reaction {
    fn split_amount_material(input: &str) -> (isize, String) {
        let splitted: Vec<&str> = input.split(' ').collect();
        assert_eq!(splitted.len(), 2);
        (splitted[0].parse::<isize>().unwrap(), splitted[1].to_string())
    }

    fn parse(input: &str) -> Reaction {
        let reaction: Vec<&str> = input.split(" => ").collect();
        let (amount, result) = Reaction::split_amount_material(reaction[1]);
        let required = reaction[0].split(", ");
        let mut resources = HashMap::new();
        for resource in required {
            let (amount, material) = Reaction::split_amount_material(resource);
            resources.insert(material, amount);
        }

        Reaction {
            result,
            amount,
            resources,
        }
    }
}

struct Reactions {
    reactions: HashMap<String, Reaction>,
}

impl Reactions {
    const ORE_LIMIT: isize = 1_000_000_000_000;

    fn parse(input: &str) -> Reactions {
        let mut reactions = HashMap::new();
        let input = input.trim_end();

        for line in input.split('\n') {
            let reaction = Reaction::parse(line);
            reactions.insert(reaction.result.clone(), reaction);
        }
        Reactions { reactions }
    }

    fn count_material_(&self, amount_requested: isize, output: &str, leftovers: &mut HashMap<String, isize>, total_ore_count: &mut isize) -> Result<isize, ()> {
        if output == "ORE" {
            if *total_ore_count + amount_requested > Reactions::ORE_LIMIT {
                return Err(());
            }
            *total_ore_count += amount_requested;
            return Ok(amount_requested);
        }
        let mut amount_requested = amount_requested;

        /* use up leftover materials first */
        let leftover = *leftovers.entry(output.to_string()).or_insert(0);
        if leftover >= amount_requested {
            leftovers.insert(output.to_string(), leftover - amount_requested);
            /* no additional reaction needed, can satisfy with leftovers */
            return Ok(0);
        } else {
            leftovers.insert(output.to_string(), 0);
            amount_requested -= leftover;
        }

        let reaction = self.reactions.get(output).unwrap();
        let mut factor = amount_requested / reaction.amount;
        if amount_requested % reaction.amount > 0 {
            factor += 1;
        }
        let produced = factor * reaction.amount;
        leftovers.insert(output.to_string(), produced - amount_requested);

        let mut ore_count = 0;
        for resource in reaction.resources.keys() {
            let amount = *reaction.resources.get(resource).unwrap();
            ore_count += self.count_material_(amount * factor, &resource, leftovers, total_ore_count)?;
        }

        Ok(ore_count)
    }

    fn count_material(&self, amount_requested: isize, output: &str, ) -> isize {
        let mut ore_count = 0;
        self.count_material_(amount_requested, output, &mut HashMap::new(), &mut ore_count).unwrap()
    }

    fn possible_fuel(&self) -> isize {
        let mut leftovers = HashMap::new();
        let mut ore_count = 0;

        let mut fuel_count = 1;
        let ore_per_fuel = self.count_material_(1, "FUEL", &mut leftovers, &mut ore_count).unwrap();

        fuel_count += Reactions::ORE_LIMIT / ore_per_fuel;
        self.count_material_(Reactions::ORE_LIMIT / ore_per_fuel, "FUEL", &mut leftovers, &mut ore_count).unwrap();
        while let Ok(_) = self.count_material_(1, "FUEL", &mut leftovers, &mut ore_count) {
            fuel_count += 1;
        }
        fuel_count
    }

}

fn day14() {
    let input = read_file("input14");
    let reactions = Reactions::parse(&input);

    println!("14a: {}", reactions.count_material(1, "FUEL"));

    println!("14b: {}", reactions.possible_fuel());
}

#[derive(Copy, Clone, PartialEq)]
#[repr(isize)]
enum DroidDirection {
    NORTH = 1,
    SOUTH = 2,
    WEST = 3,
    EAST = 4,
}

impl DroidDirection {
    fn from(val: isize) -> Option<DroidDirection> {
        match val {
            1 => Some(DroidDirection::NORTH),
            2 => Some(DroidDirection::SOUTH),
            3 => Some(DroidDirection::WEST),
            4 => Some(DroidDirection::EAST),
            _ => None,
        }
    }

    fn opposite(self) -> DroidDirection {
        match self {
            DroidDirection::NORTH => DroidDirection::SOUTH,
            DroidDirection::SOUTH => DroidDirection::NORTH,
            DroidDirection::WEST => DroidDirection::EAST,
            DroidDirection::EAST => DroidDirection::WEST,
        }
    }

    fn next(self) -> Option<DroidDirection> {
        match self {
            DroidDirection::NORTH => Some(DroidDirection::SOUTH),
            DroidDirection::SOUTH => Some(DroidDirection::WEST),
            DroidDirection::WEST => Some(DroidDirection::EAST),
            DroidDirection::EAST => None,
        }
    }
}

struct RepairDroid {
    computer: IntComputer,
    steps: u32,
    pos: Point,
    map: HashMap<Point, isize>,
    direction: DroidDirection,
    path: Vec<(Point, DroidDirection)>,
}

impl RepairDroid {
    fn new(program: &[isize]) -> RepairDroid {
        let mut tiles_map_ascii = HashMap::new();
        tiles_map_ascii.insert(0, ' ');
        tiles_map_ascii.insert(1, '.');
        tiles_map_ascii.insert(2, '#');
        tiles_map_ascii.insert(3, 'O');
        tiles_map_ascii.insert(4, 'D');
        tiles_map_ascii.insert(5, '+');

        let mut tiles_map_color = HashMap::new();
        tiles_map_color.insert(0, (255, 255, 255));
        tiles_map_color.insert(1, (200, 200, 200));
        tiles_map_color.insert(2, (  0,   0,   0));
        tiles_map_color.insert(3, (  0,   0, 255));
        tiles_map_color.insert(4, (  0, 255,   0));
        tiles_map_color.insert(5, (150, 150, 150));

        let mut computer = IntComputer::new(&program);
        computer.screen.tiles_map_ascii = tiles_map_ascii;
        computer.screen.tiles_map_color = tiles_map_color;

        RepairDroid {
            computer,
            steps: 0,
            pos: Point { x: 0, y: 0 },
            map: HashMap::new(),
            direction: DroidDirection::NORTH,
            path: Vec::new(),
        }
    }

    fn move_direction(&mut self) {
        let mut new_pos = self.pos;
        match self.direction {
            DroidDirection::NORTH => new_pos.y += 1,
            DroidDirection::SOUTH => new_pos.y -= 1,
            DroidDirection::WEST  => new_pos.x -= 1,
            DroidDirection::EAST  => new_pos.x += 1,
        }
        self.map.insert(new_pos, 1);

        /* start next movement to north, except we are coming from there */
        let old_direction = self.direction;
        self.direction = if self.direction == DroidDirection::SOUTH {
            DroidDirection::SOUTH
        } else {
            DroidDirection::NORTH
        };

        let mut drop_last = false;
        if let Some((pos, _)) = self.path.last() {
            /* if the new pos is the same as the previous
               on our path, we are tracking back */
            if self.pos != *pos && new_pos == *pos {
                self.direction = old_direction.opposite();
                drop_last = true;
                self.direction = self.next_direction();
            }
        }
        if drop_last {
            self.pos = new_pos;
            return;
        }

        /* save the successful movement */
        self.pos = new_pos;
        self.path.push((self.pos, old_direction));
    }

    fn mark_wall(&mut self) {
        let mut wall = self.pos;
        match self.direction {
            DroidDirection::NORTH => wall.y += 1,
            DroidDirection::SOUTH => wall.y -= 1,
            DroidDirection::WEST  => wall.x -= 1,
            DroidDirection::EAST  => wall.x += 1,
        }
        self.map.insert(wall, 2);
    }

    fn draw_map(&mut self) {
        for (pos, val) in &self.map {
            self.computer.screen.set_pixel(pos.x as isize, pos.y as isize, *val);
        }
        self.computer.screen.set_pixel(0, 0, 5);
        self.computer.screen.set_pixel(self.pos.x as isize, self.pos.y as isize, 4);
        self.computer.screen.dump_screen(None);
    }

    fn next_direction(&mut self) -> DroidDirection {
        let previous_direction = match self.path.last() {
            None => DroidDirection::NORTH,
            Some(d) => d.1,
        };

        let mut new_direction = self.direction.next();
        if let Some(dir) = new_direction {
            if dir == previous_direction.opposite() {
                new_direction = dir.next();
            }
        }

        if let Some(d) = new_direction {
            d
        } else {
            /* go back */
            self.path.pop();
            previous_direction.opposite()
        }
    }

    fn set_to_oxygen(&mut self, x: i32, y: i32) {
        if let Some(1) = self.map.get(&Point {x, y}) {
            self.map.insert(Point {x, y}, 3);
        }
    }

    fn fill_with_oxygen(&mut self) -> u32 {
        self.solve_labyrinth(true);

        let mut count = 0;
        while self.map.values().filter(|&x| *x == 1).count() > 0 {
            let oxygens: Vec<Point> = self.map.iter()
                                              .filter(|(_, &val)| val == 3)
                                              .map(|(&pos, _)| pos)
                                              .collect();

            for pos in oxygens {
                self.set_to_oxygen(pos.x + 1, pos.y);
                self.set_to_oxygen(pos.x - 1, pos.y);
                self.set_to_oxygen(pos.x, pos.y + 1);
                self.set_to_oxygen(pos.x, pos.y - 1);
            }
            count += 1;
        }
        count
    }

    fn repair(&mut self) {
        self.solve_labyrinth(false);
    }

    fn solve_labyrinth(&mut self, map_completely: bool) {
        loop {
            match self.computer.run_program() {
                ProgramState::NEEDINPUT => {
                    self.computer.input.push_back(self.direction as isize);
                }
                ProgramState::SUSPENDED => {
                    match self.computer.output.pop_front().unwrap() {
                        0 => { /* hit a wall */
                            self.mark_wall();
                            self.direction = self.next_direction();
                        }
                        1 => { /* moving in this direction was successful */
                            self.move_direction();
                            if self.path.is_empty() {
                                break;
                            }
                        }
                        2 => { /* found the destination */
                            self.move_direction();
                            self.map.insert(self.pos, 3);
                            if !map_completely {
                                break;
                            }
                        }
                        _ => panic!("invalid droid state")
                    }
                }
                ProgramState::HALTED => break,
                _ => {}
            }
        }
    }
}

fn day15() {
    let input = read_file("input15");
    let input = input.trim_end();
    let program : Vec<isize> = input.split(',')
                                    .map(|x| x.parse::<isize>().unwrap())
                                    .collect();

    let mut droid = RepairDroid::new(&program);
    droid.computer.suspend_on_output = true;
    droid.repair();
    println!("15a: {}", droid.path.len());

    let mut droid = RepairDroid::new(&program);
    droid.computer.suspend_on_output = true;
    let time = droid.fill_with_oxygen();
    println!("15b: {}", time);
}

fn fft_pattern(i: usize, j: usize) -> i8 {
    if j < i {
        return 0;
    }
    let len = 4 * (i + 1);

    let offset = (j+1) % len;
    if offset < (i+1) {
        return 0;
    } else if offset < ((i+1) * 2) {
        return 1;
    } else if offset < ((i+1) * 3) {
        return 0;
    } else if offset < ((i+1) * 4) {
        return -1;
    }
    panic!("invalid offset");
}

fn fft(input: &[i8]) -> Vec<i8> {
    let mut output = vec![0; input.len()/2];

    for i in 0..input.len()/2 {
        let mut sum : i32 = 0;
        for j in i..input.len() {
            let factor = fft_pattern(i, j);
            match factor {
                0 => continue,
                1 => sum += input[j] as i32,
               -1 => sum -= input[j] as i32,
                _ => panic!("unexpected fft pattern value")
            }
        }
        output[i] = (sum % 10).abs() as i8;
    }
    let mut half2 = fft_2nd_half(&input).split_off(input.len()/2);
    output.append(&mut half2);

    output
}

fn fft_2nd_half(input: &[i8]) -> Vec<i8> {
    let mut output = vec![0; input.len()];

    let mut sum = 0;
    for i in (input.len()/2..input.len()).rev() {
        sum += input[i];
        sum %= 10;
        output[i] = sum;
    }

    output
}

fn day16() {
    let input = read_file("input16");
    let input = input.trim_end();
    let mut values : Vec<i8> = input.chars().map(|x| x.to_digit(10).unwrap() as i8).collect();
    for _ in 0..100 {
        values = fft(&values);
    }

    print!("16a: ");
    for d in values.iter().take(8) {
        print!("{}", d);
    }
    println!();

    let mut long_input = String::new();
    long_input.reserve(input.len() * 10_000);
    for _ in 0..10_000 {
        long_input += input;
    }
    let mut values : Vec<i8> = long_input.chars().map(|x| x.to_digit(10).unwrap() as i8).collect();
    let mut pos : usize = 0;
    for i in values.iter().take(7) {
        pos *= 10;
        pos += *i as usize;
    }
    for _ in 0..100 {
        values = fft_2nd_half(&values);
    }
    print!("16b: ");
    for d in values.iter().skip(pos).take(8) {
        print!("{}", d);
    }
    println!();
}

fn is_scaffold(map: &[Vec<char>], pos: Point) -> bool {
    if pos.x < 0 || pos.y < 0 || pos.y >= map.len() as i32 || pos.x >= map[pos.y as usize].len() as i32 {
        false
    } else {
        map[pos.y as usize][pos.x as usize] == '#'
    }
}

fn can_move_forward(map: &[Vec<char>], pos: &mut Point, direction: Direction) -> bool {
    let new_pos = match direction {
        Direction::UP => Point {x: pos.x, y: pos.y-1},
        Direction::DOWN => Point {x: pos.x, y: pos.y+1},
        Direction::LEFT => Point {x: pos.x-1, y: pos.y},
        Direction::RIGHT => Point {x: pos.x+1, y: pos.y},
    };
    if is_scaffold(&map, new_pos) {
        *pos = new_pos;
        true
    } else {
        false
    }
}

fn turn_until_aligned(map: &[Vec<char>], pos: Point, direction: &mut Direction) -> Option<Direction> {
    let old_direction = *direction;
    match old_direction {
        Direction::DOWN | Direction::UP => {
            if is_scaffold(&map, Point {x: pos.x-1, y: pos.y}) {
                *direction = Direction::LEFT;
                if old_direction == Direction::DOWN { Some(Direction::RIGHT) } else { Some(Direction::LEFT) }
            } else if is_scaffold(&map, Point {x: pos.x+1, y: pos.y}) {
                *direction = Direction::RIGHT;
                if old_direction == Direction::DOWN { Some(Direction::LEFT) } else { Some(Direction::RIGHT) }
            } else {
                None
            }
        }
        Direction::LEFT | Direction::RIGHT => {
            if is_scaffold(&map, Point {x: pos.x, y: pos.y-1}) {
                *direction = Direction::UP;
                if old_direction == Direction::LEFT { Some(Direction::RIGHT) } else { Some(Direction::LEFT) }
            } else if is_scaffold(&map, Point {x: pos.x, y: pos.y+1}) {
                *direction = Direction::DOWN;
                if old_direction == Direction::LEFT { Some(Direction::LEFT) } else { Some(Direction::RIGHT) }
            } else {
                None
            }
        }
    }
}

fn get_path(map: &[Vec<char>], start: Point) -> Vec<String> {
    let mut path = Vec::new();
    let mut direction = match map[start.y as usize][start.x as usize] {
        '^' => Direction::UP,
        'v' => Direction::DOWN,
        '<' => Direction::LEFT,
        '>' => Direction::RIGHT,
        _ => panic!("invalid direction")
    };
    let mut pos = start;
    loop {
        let mut forward = 0;
        while can_move_forward(&map, &mut pos, direction) {
            forward += 1;
        }
        if forward > 0 {
            path.push(forward.to_string());
        }
        let rotate = turn_until_aligned(&map, pos, &mut direction);
        let rot_str = match rotate {
            None => break,
            Some(Direction::LEFT) => "L",
            Some(Direction::RIGHT) => "R",
            _ => panic!("invalid rotation")
        };
        path.push(rot_str.to_string());
    }
    path
}

fn day17() {
    let input = read_file("input17");
    let input = input.trim_end();
    let program : Vec<isize> = input.split(',')
                                    .map(|x| x.parse::<isize>().unwrap())
                                    .collect();

    let mut computer = IntComputer::new(&program);
    computer.suspend_on_output = true;

    let mut map = Vec::new();
    let mut row = Vec::new();
    loop {
        match computer.run_program() {
            ProgramState::SUSPENDED => {
                let output = (computer.output.pop_front().unwrap() as u8) as char;
                if output == '\n' {
                    map.push(row.clone());
                    row.clear();
                } else {
                    row.push(output);
                }
            }
            ProgramState::HALTED => break,
            _ => {}
        }
    }
    let mut start = Point {x: 0, y: 0};
    let mut sum = 0;
    for y in 0..map.len() {
        for x in 0..map[y].len() {
            if map[y][x] == '^' || map[y][x] == 'v' || map[y][x] == '>' || map[y][x] == '<' {
                start = Point {x: x as i32, y: y as i32};
            }
            if map[y][x] != '#' || x < 1 || y < 1 || x >= map[y].len()-1 || y >= map.len()-1 {
                continue;
            }
            if map[y][x-1] == '#' && map[y][x+1] == '#' && map[y-1][x] == '#' && map[y+1][x] == '#' {
                sum += x * y;
            }
        }
    }

    println!("17a: {}", sum);

    let _path = get_path(&map, start);

    // resulting path is: R,10,L,12,R,6,R,10,L,12,R,6,R,6,R,10,R,12,R,6,R,10,L,12,L,12,R,6,R,10,R,12,R,6,R,10,L,12,L,12,R,6,R,10,R,12,R,6,R,10,L,12,L,12,R,6,R,10,R,12,R,6,R,10,L,12,R,6
    // partitions have been manually determined

    let mut input = String::new();
    input += "A,A,B,C,B,C,B,C,B,A\n"; // main routine;
    input += "R,10,L,12,R,6\n"; // A
    input += "R,6,R,10,R,12,R,6\n"; // B;
    input += "R,10,L,12,L,12\n"; // C;
    input += "n\n";

    let mut computer = IntComputer::new(&program);
    computer.mem[0] = 2;
    computer.input = input.bytes().map(|i| i as isize).collect();
    computer.run_program();

    println!("17b: {}", computer.output.pop_back().unwrap());
}

fn check_position(program: &[isize], x: isize, y: isize) -> isize {
    let mut computer = IntComputer::new(program);
    computer.input.push_back(x);
    computer.input.push_back(y);
    computer.run_program();
    computer.output.pop_front().unwrap()
}

fn check_square_fits(map: &HashSet<(isize, isize)>, size: isize, x: isize, y: isize) -> bool {
    map.contains(&(x, y-size+1)) &&
    map.contains(&(x-size+1, y)) &&
    map.contains(&(x-size+1, y-size+1)) &&
    map.contains(&(x, y))
}

fn day19() {
    let input = read_file("input19");
    let input = input.trim_end();
    let program : Vec<isize> = input.split(',')
                                    .map(|x| x.parse::<isize>().unwrap())
                                    .collect();

    let mut sum = 0;
    for x in 0..50 {
        for y in 0..50 {
            sum += check_position(&program, x, y);
        }
    }
    println!("19a: {}", sum);

    let mut map = HashSet::new();
    let mut tractor_begin = 0;
    'outer: for y in 0..1500 {
        let mut was_tractor = false;
        for x in tractor_begin..1500 {
            let state = check_position(&program, x, y);
            if state == 0 {
                if was_tractor {
                    /* any further position on the x axis will be space */
                    break;
                }
            } else {
                if !was_tractor {
                    was_tractor = true;
                    tractor_begin = x;
                }
                map.insert((x, y));
                if check_square_fits(&map, 100, x, y) {
                    println!("19b: {}", (x-99)*10_000 + (y-99));
                    break 'outer;
                }
            }
            sum += state;
        }
    }
}

fn day21() {
    let input = read_file("input21");
    let input = input.trim_end();
    let program: Vec<isize> = input.split(',')
                                   .map(|x| x.parse::<isize>().unwrap())
                                   .collect();

    // !A OR (!C AND D)
    let script = "NOT A J\n\
                  NOT C T\n\
                  AND D T\n\
                  OR T J\n\
                  WALK\n";

    let mut computer = IntComputer::new(&program);
    computer.input = script.bytes().map(|b| b as isize).collect();
    computer.run_program();
    if *computer.output.back().unwrap() > 127 {
        println!("21a: {}", computer.output.back().unwrap());
    } else {
        for c in computer.output {
            print!("{}", (c as u8) as char);
        }
    }

    // (!A OR !B OR !C) AND D AND H OR !A
    let script = "NOT A J\n\
                  NOT B T\n\
                  OR T J\n\
                  NOT C T\n\
                  OR T J\n\
                  AND D J\n\
                  AND H J\n\
                  NOT A T\n\
                  OR T J\n\
                  RUN\n";

    let mut computer = IntComputer::new(&program);
    computer.input = script.bytes().map(|b| b as isize).collect();
    computer.run_program();

    if *computer.output.back().unwrap() > 127 {
        println!("21b: {}", computer.output.back().unwrap());
    } else {
        for c in computer.output {
            print!("{}", (c as u8) as char);
        }
    }
}

#[derive(Clone, Copy)]
enum ShuffleTechnique {
    DealInto,
    CutN { n: i64 },
    DealInc { n: u128 },
}

impl ShuffleTechnique {
    fn parse(input: &str) -> ShuffleTechnique {
        if input == "deal into new stack" {
            ShuffleTechnique::DealInto
        } else if input.find("cut").is_some() {
            let n = input.split_at(4).1.parse::<i64>().unwrap();
            ShuffleTechnique::CutN { n }
        }
        else if input.find("deal with increment").is_some() {
            let n = input.split_at(20).1.parse::<u128>().unwrap();
            ShuffleTechnique::DealInc { n }
        }
        else {
            panic!("unsupported technique")
        }
    }
}

struct SpaceCards {
    deck: VecDeque<usize>,
}

impl SpaceCards {
    fn new(size: usize) -> SpaceCards {
        SpaceCards {
            deck: (0..size).collect()
        }
    }

    fn shuffle(&mut self, technique: ShuffleTechnique) {
        match technique {
            ShuffleTechnique::DealInto => {
                self.deck = self.deck.iter().rev().copied().collect();
            },
            ShuffleTechnique::CutN { n } => {
                if n >= 0 {
                    self.deck.rotate_left(n as usize);
                } else {
                    self.deck.rotate_right(n.abs() as usize);
                }
            },
            ShuffleTechnique::DealInc { n } => {
                let mut deck = VecDeque::new();
                deck.resize(self.deck.len(), 0);
                let mut pos = 0;
                for c in self.deck.iter() {
                    deck[pos] = *c;
                    pos += n as usize;
                    pos %= self.deck.len();
                }
                self.deck = deck;
            },
        }
    }

    fn unshuffle(size: u128, pos: u128, technique: ShuffleTechnique) -> u128 {
        match technique {
            ShuffleTechnique::DealInto => {
                size - pos - 1
            },
            ShuffleTechnique::CutN { n } => {
                let shift = if n < 0 {
                    size - (n.abs() as u128)
                } else {
                    n as u128
                };
                (pos + shift) % size
            },
            ShuffleTechnique::DealInc { n } => {
                // n*x = t*size + pos
                // x = (t*size + pos) / n
                let mut x = None;
                for t in 0..n {
                    let tmp = t * size + pos;
                    if tmp % n == 0 {
                        x = Some(tmp / n);
                        break;
                    }
                }
                x.unwrap()
            },
        }
    }
}

fn day22() {
    let input = read_file("input22");
    let input = input.trim_end();
    let shuffles: Vec<ShuffleTechnique> = input.split('\n')
                                               .map(|t| ShuffleTechnique::parse(t))
                                               .collect();

    let mut cards = SpaceCards::new(10007);
    for technique in &shuffles {
        cards.shuffle(*technique);
    }
    println!("22a: {}", cards.deck.iter().position(|&c| c == 2019).unwrap());
}

fn main() {
    day22();
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

    #[test]
    fn test_day9() {
        let program = vec![109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99];
        let mut computer = IntComputer::new(&program);
        computer.run_program();
        assert!(computer.output.iter().eq(program.iter()));

        let program = vec![1102,34915192,34915192,7,4,7,99,0];
        let mut computer = IntComputer::new(&program);
        computer.run_program();
        assert_eq!(computer.output[0], 1219070632396864);

        let program = vec![104,1125899906842624,99];
        let mut computer = IntComputer::new(&program);
        computer.run_program();
        assert_eq!(computer.output[0], 1125899906842624);
    }

    #[test]
    fn test_day10() {
        let map = ".#..#\n\
                   .....\n\
                   #####\n\
                   ....#\n\
                   ...##";
        let asteroid_map = AsteroidMap::new(map);
        assert_eq!(asteroid_map.max_detected_asteroids().1, 8);

        let map = "......#.#.\n\
                   #..#.#....\n\
                   ..#######.\n\
                   .#.#.###..\n\
                   .#..#.....\n\
                   ..#....#.#\n\
                   #..#....#.\n\
                   .##.#..###\n\
                   ##...#..#.\n\
                   .#....####";
        let asteroid_map = AsteroidMap::new(map);
        assert_eq!(asteroid_map.max_detected_asteroids().1, 33);

        let map = "#.#...#.#.\n\
                   .###....#.\n\
                   .#....#...\n\
                   ##.#.#.#.#\n\
                   ....#.#.#.\n\
                   .##..###.#\n\
                   ..#...##..\n\
                   ..##....##\n\
                   ......#...\n\
                   .####.###.";
        let asteroid_map = AsteroidMap::new(map);
        assert_eq!(asteroid_map.max_detected_asteroids().1, 35);

        let map = ".#..#..###\n\
                   ####.###.#\n\
                   ....###.#.\n\
                   ..###.##.#\n\
                   ##.##.#.#.\n\
                   ....###..#\n\
                   ..#.#..#.#\n\
                   #..#.#.###\n\
                   .##...##.#\n\
                   .....#.#..";
        let asteroid_map = AsteroidMap::new(map);
        assert_eq!(asteroid_map.max_detected_asteroids().1, 41);

        let map = ".#..##.###...#######\n\
                   ##.############..##.\n\
                   .#.######.########.#\n\
                   .###.#######.####.#.\n\
                   #####.##.#.##.###.##\n\
                   ..#####..#.#########\n\
                   ####################\n\
                   #.####....###.#.#.##\n\
                   ##.#################\n\
                   #####.##.###..####..\n\
                   ..######..##.#######\n\
                   ####.##.####...##..#\n\
                   .#####..#.######.###\n\
                   ##...#.##########...\n\
                   #.##########.#######\n\
                   .####.#.###.###.#.##\n\
                   ....##.##.###..#####\n\
                   .#.#.###########.###\n\
                   #.#.#.#####.####.###\n\
                   ###.##.####.##.#..##";
        let mut asteroid_map = AsteroidMap::new(map);
        let max = asteroid_map.max_detected_asteroids();
        assert_eq!(max.1, 210);

        assert_eq!(asteroid_map.vaporized(max.0, 200), Point { x: 8, y: 2 });
    }

    #[test]
    fn test_day14() {
        let input = "10 ORE => 10 A\n\
                     1 ORE => 1 B\n\
                     7 A, 1 B => 1 C\n\
                     7 A, 1 C => 1 D\n\
                     7 A, 1 D => 1 E\n\
                     7 A, 1 E => 1 FUEL";
        let reactions = Reactions::parse(input);
        assert_eq!(reactions.count_material(1, "FUEL"), 31);

        let input = "9 ORE => 2 A\n\
                     8 ORE => 3 B\n\
                     7 ORE => 5 C\n\
                     3 A, 4 B => 1 AB\n\
                     5 B, 7 C => 1 BC\n\
                     4 C, 1 A => 1 CA\n\
                     2 AB, 3 BC, 4 CA => 1 FUEL";
        let reactions = Reactions::parse(input);
        assert_eq!(reactions.count_material(1, "FUEL"), 165);


        let input = "157 ORE => 5 NZVS\n\
                     165 ORE => 6 DCFZ\n\
                     44 XJWVT, 5 KHKGT, 1 QDVJ, 29 NZVS, 9 GPVTF, 48 HKGWZ => 1 FUEL\n\
                     12 HKGWZ, 1 GPVTF, 8 PSHF => 9 QDVJ\n\
                     179 ORE => 7 PSHF\n\
                     177 ORE => 5 HKGWZ\n\
                     7 DCFZ, 7 PSHF => 2 XJWVT\n\
                     165 ORE => 2 GPVTF\n\
                     3 DCFZ, 7 NZVS, 5 HKGWZ, 10 PSHF => 8 KHKGT";
        let reactions = Reactions::parse(input);
        assert_eq!(reactions.count_material(1, "FUEL"), 13312);
        //assert_eq!(reactions.possible_fuel(), 82892753);

        let input = "2 VPVL, 7 FWMGM, 2 CXFTF, 11 MNCFX => 1 STKFG\n\
                     17 NVRVD, 3 JNWZP => 8 VPVL\n\
                     53 STKFG, 6 MNCFX, 46 VJHF, 81 HVMC, 68 CXFTF, 25 GNMV => 1 FUEL\n\
                     22 VJHF, 37 MNCFX => 5 FWMGM\n\
                     139 ORE => 4 NVRVD\n\
                     144 ORE => 7 JNWZP\n\
                     5 MNCFX, 7 RFSQX, 2 FWMGM, 2 VPVL, 19 CXFTF => 3 HVMC\n\
                     5 VJHF, 7 MNCFX, 9 VPVL, 37 CXFTF => 6 GNMV\n\
                     145 ORE => 6 MNCFX\n\
                     1 NVRVD => 8 CXFTF\n\
                     1 VJHF, 6 MNCFX => 4 RFSQX\n\
                     176 ORE => 6 VJHF";
        let reactions = Reactions::parse(input);
        assert_eq!(reactions.count_material(1, "FUEL"), 180697);
        //assert_eq!(reactions.possible_fuel(), 5586022);

        let input = "171 ORE => 8 CNZTR\n\
                     7 ZLQW, 3 BMBT, 9 XCVML, 26 XMNCP, 1 WPTQ, 2 MZWV, 1 RJRHP => 4 PLWSL\n\
                     114 ORE => 4 BHXH\n\
                     14 VRPVC => 6 BMBT\n\
                     6 BHXH, 18 KTJDG, 12 WPTQ, 7 PLWSL, 31 FHTLT, 37 ZDVW => 1 FUEL\n\
                     6 WPTQ, 2 BMBT, 8 ZLQW, 18 KTJDG, 1 XMNCP, 6 MZWV, 1 RJRHP => 6 FHTLT\n\
                     15 XDBXC, 2 LTCX, 1 VRPVC => 6 ZLQW\n\
                     13 WPTQ, 10 LTCX, 3 RJRHP, 14 XMNCP, 2 MZWV, 1 ZLQW => 1 ZDVW\n\
                     5 BMBT => 4 WPTQ\n\
                     189 ORE => 9 KTJDG\n\
                     1 MZWV, 17 XDBXC, 3 XCVML => 2 XMNCP\n\
                     12 VRPVC, 27 CNZTR => 2 XDBXC\n\
                     15 KTJDG, 12 BHXH => 5 XCVML\n\
                     3 BHXH, 2 VRPVC => 7 MZWV\n\
                     121 ORE => 7 VRPVC\n\
                     7 XCVML => 6 RJRHP\n\
                     5 BHXH, 4 VRPVC => 5 LTCX";
        let reactions = Reactions::parse(input);
        assert_eq!(reactions.count_material(1, "FUEL"), 2210736);
        //assert_eq!(reactions.possible_fuel(), 460664);
    }

    #[test]
    fn test_day16() {
        assert_eq!(fft(&[1,2,3,4,5,6,7,8]), [4,8,2,2,6,1,5,8]);
        assert_eq!(fft(&[4,8,2,2,6,1,5,8]), [3,4,0,4,0,4,3,8]);
        assert_eq!(fft(&[3,4,0,4,0,4,3,8]), [0,3,4,1,5,5,1,8]);
        assert_eq!(fft(&[0,3,4,1,5,5,1,8]), [0,1,0,2,9,4,9,8]);

        let input = "80871224585914546619083218645595";
        let mut values : Vec<i8> = input.chars().map(|x| x.to_digit(10).unwrap() as i8).collect();
        for _ in 0..100 {
            values = fft(&values);
        }
        assert_eq!(values[0..8], [2,4,1,7,6,1,7,6]);

        let input = "19617804207202209144916044189917";
        let mut values : Vec<i8> = input.chars().map(|x| x.to_digit(10).unwrap() as i8).collect();
        for _ in 0..100 {
            values = fft(&values);
        }
        assert_eq!(values[0..8], [7,3,7,4,5,4,1,8]);

        let input = "69317163492948606335995924319873";
        let mut values : Vec<i8> = input.chars().map(|x| x.to_digit(10).unwrap() as i8).collect();
        for _ in 0..100 {
            values = fft(&values);
        }
        assert_eq!(values[0..8], [5,2,4,3,2,1,3,3]);
    }

    #[test]
    fn test_day22() {
        let techniques = vec![
            ShuffleTechnique::parse("deal with increment 7"),
            ShuffleTechnique::parse("deal into new stack"),
            ShuffleTechnique::parse("deal into new stack"),
        ];
        let mut cards = SpaceCards::new(10);

        for technique in techniques {
            cards.shuffle(technique);
        }
        assert_eq!(cards.deck, [0,3,6,9,2,5,8,1,4,7]);

        let techniques = vec![
            ShuffleTechnique::parse("cut 6"),
            ShuffleTechnique::parse("deal with increment 7"),
            ShuffleTechnique::parse("deal into new stack"),
        ];
        let mut cards = SpaceCards::new(10);
        for technique in techniques {
            cards.shuffle(technique);
        }
        assert_eq!(cards.deck, [3,0,7,4,1,8,5,2,9,6]);

        let techniques = vec![
            ShuffleTechnique::parse("deal into new stack"),
            ShuffleTechnique::parse("cut -2"),
            ShuffleTechnique::parse("deal with increment 7"),
            ShuffleTechnique::parse("cut 8"),
            ShuffleTechnique::parse("cut -4"),
            ShuffleTechnique::parse("deal with increment 7"),
            ShuffleTechnique::parse("cut 3"),
            ShuffleTechnique::parse("deal with increment 9"),
            ShuffleTechnique::parse("deal with increment 3"),
            ShuffleTechnique::parse("cut -1"),
        ];
        let mut cards = SpaceCards::new(10);
        for technique in techniques {
            cards.shuffle(technique);
        }
        assert_eq!(cards.deck, [9,2,5,8,1,4,7,0,3,6]);


        assert_eq!(SpaceCards::unshuffle(10, 2, ShuffleTechnique::DealInto), 7);
        assert_eq!(SpaceCards::unshuffle(10, 8, ShuffleTechnique::DealInto), 1);

        assert_eq!(SpaceCards::unshuffle(10, 0, ShuffleTechnique::CutN {n: 3}), 3);
        assert_eq!(SpaceCards::unshuffle(10, 3, ShuffleTechnique::CutN {n: 3}), 6);
        assert_eq!(SpaceCards::unshuffle(10, 0, ShuffleTechnique::CutN {n: -4}), 6);
        assert_eq!(SpaceCards::unshuffle(10, 8, ShuffleTechnique::CutN {n: -4}), 4);

        assert_eq!(SpaceCards::unshuffle(10, 0, ShuffleTechnique::DealInc {n: 3}), 0);
        assert_eq!(SpaceCards::unshuffle(10, 3, ShuffleTechnique::DealInc {n: 3}), 1);
        assert_eq!(SpaceCards::unshuffle(10, 6, ShuffleTechnique::DealInc {n: 3}), 2);
        assert_eq!(SpaceCards::unshuffle(10, 9, ShuffleTechnique::DealInc {n: 3}), 3);
        assert_eq!(SpaceCards::unshuffle(10, 2, ShuffleTechnique::DealInc {n: 3}), 4);
        assert_eq!(SpaceCards::unshuffle(10, 7, ShuffleTechnique::DealInc {n: 3}), 9);
    }
}
