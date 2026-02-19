//! A string utilities library providing various string manipulation functions.
//!
//! This library contains helper functions for common string operations
//! with a focus on performance and correctness.

/// Reverses the input string while properly handling UTF-8 characters.
///
/// This function takes a string slice and returns a new String containing
/// the characters in reverse order. It correctly handles multi-byte UTF-8
/// characters by reversing at the character boundary level rather than
/// at the byte level.
///
/// # Arguments
///
/// * `input` - A string slice to be reversed
///
/// # Returns
///
/// A new `String` containing the reversed content
///
/// # Examples
///
/// ```
/// use stringutils::reverse_string;
///
/// let result = reverse_string("hello");
/// assert_eq!(result, "olleh");
///
/// let result = reverse_string("Hello, 世界!");
/// assert_eq!(result, "!界世 ,olleH");
/// ```
pub fn reverse_string(input: &str) -> String {
    input.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_empty_string() {
        assert_eq!(reverse_string(""), "");
    }

    #[test]
    fn test_reverse_simple_string() {
        assert_eq!(reverse_string("hello"), "olleh");
    }

    #[test]
    fn test_reverse_single_char() {
        assert_eq!(reverse_string("a"), "a");
    }

    #[test]
    fn test_reverse_with_spaces() {
        assert_eq!(reverse_string("hello world"), "dlrow olleh");
    }

    #[test]
    fn test_reverse_unicode() {
        assert_eq!(reverse_string("Hello, 世界!"), "!界世 ,olleH");
    }

    #[test]
    fn test_reverse_emoji() {
        assert_eq!(reverse_string("Hello 👋 World 🌍"), "🌍 dlroW 👋 olleH");
    }

    #[test]
    fn test_reverse_numbers() {
        assert_eq!(reverse_string("12345"), "54321");
    }

    #[test]
    fn test_reverse_mixed_content() {
        let input = "The quick brown fox jumps over the lazy dog";
        let expected = "god yzal eht revo spmuj xof nworb kciuq ehT";
        assert_eq!(reverse_string(input), expected);
    }
}