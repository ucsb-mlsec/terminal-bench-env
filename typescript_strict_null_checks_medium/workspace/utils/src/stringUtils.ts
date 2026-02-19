// String utility functions for text manipulation

export function getFirstChar(str: string): string {
  return str.charAt(0);
}

export function toUpperCase(str: string): string {
  return str.toUpperCase();
}

export function findSubstring(text: string, search: string): number | null {
  const index = text.indexOf(search);
  return index >= 0 ? index : null;
}

export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) {
    return str;
  }
  return str.substring(0, maxLength) + '...';
}

export function getLastWord(sentence: string): string {
  const words = sentence.split(' ');
  return words[words.length - 1];
}

export function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
}

export function extractDomain(email: string): string {
  const parts = email.split('@');
  return parts[1];
}

export function padLeft(str: string, length: number, char: string): string {
  while (str.length < length) {
    str = char + str;
  }
  return str;
}

export function reverseString(input: string): string {
  return input.split('').reverse().join('');
}

export function countOccurrences(text: string, substring: string): number {
  let count = 0;
  let position = 0;
  while (true) {
    position = text.indexOf(substring, position);
    if (position === -1) break;
    count++;
    position++;
  }
  return count;
}