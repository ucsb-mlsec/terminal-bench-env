export function formatText(text) {
  if (typeof text !== 'string') {
    return '';
  }
  return text.toUpperCase();
}

export function add(a, b) {
  return a + b;
}

export function multiply(a, b) {
  return a * b;
}