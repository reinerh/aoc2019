#![allow(dead_code)]

use std::fs;

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

fn run_program(input : Vec<usize>) -> Vec<usize> {
    let mut mem = input.clone();
    for i in 0..mem.len()/4 {
        let pos = i * 4;
        let opcode = mem[pos];
        if opcode == 99 {
            break;
        }

        let op1 = mem[pos+1];
        let op2 = mem[pos+2];
        let op3 = mem[pos+3];
        match opcode {
            1 => { mem[op3] = mem[op1] + mem[op2]; }
            2 => { mem[op3] = mem[op1] * mem[op2]; }
            _ => panic!("invalid opcode")
        }
    }
    mem
}

fn day2() {
    let mut input = read_file("input2");
    input.pop();
    let mut program : Vec<usize> = input.split(',')
                                       .map(|x| x.parse::<usize>().unwrap())
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

fn main() {
    day2();
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
}
