export function helper(input: string): string {
  const uppercased = input.toUpperCase();
  const result = `PROCESSED: ${uppercased}`;
  return result;
}

export function formatDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export function validateInput(data: any): boolean {
  if (data === null || data === undefined) {
    return false;
  }
  if (typeof data === 'string' && data.trim().length === 0) {
    return false;
  }
  return true;
}

export function calculateSum(numbers: number[]): number {
  const total = numbers.reduce((acc, curr) => acc + curr, 0);
  return total;
}

export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}