use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 3 {
        eprintln!("Error: Expected exactly 2 arguments");
        eprintln!("Usage: {} <input_file> <output_file>", args[0]);
        process::exit(1);
    }
    
    let input_path = &args[1];
    let output_path = &args[2];
    
    if let Err(e) = process_files(input_path, output_path) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn process_files(input_path: &str, output_path: &str) -> io::Result<()> {
    let content = fs::read_to_string(input_path)
        .map_err(|e| io::Error::new(e.kind(), format!("Failed to read input file '{}': {}", input_path, e)))?;
    
    let word_counts = count_words(&content);
    
    let mut words: Vec<&String> = word_counts.keys().collect();
    words.sort();
    
    let mut output = String::new();
    for word in words {
        output.push_str(&format!("{}: {}\n", word, word_counts[word]));
    }
    
    fs::write(output_path, output)
        .map_err(|e| io::Error::new(e.kind(), format!("Failed to write output file '{}': {}", output_path, e)))?;
    
    Ok(())
}

fn count_words(text: &str) -> HashMap<String, usize> {
    let mut word_counts = HashMap::new();
    
    for word in text.split_whitespace() {
        let cleaned_word = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        
        if !cleaned_word.is_empty() {
            *word_counts.entry(cleaned_word).or_insert(0) += 1;
        }
    }
    
    word_counts
}