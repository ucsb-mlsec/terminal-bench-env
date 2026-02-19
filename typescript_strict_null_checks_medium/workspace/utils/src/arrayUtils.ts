export function getFirst<T>(arr: T[]): T {
  return arr[0];
}

export function getLast<T>(arr: T[]): T {
  return arr[arr.length - 1];
}

export function findById(items: Array<{id: number, name: string}>, id: number) {
  const result = items.find(item => item.id === id);
  return result.name;
}

export function sumNumbers(numbers: number[]): number {
  return numbers.reduce((sum, num) => sum + num, 0);
}

export function getSecondElement<T>(arr: T[]): T {
  return arr[1];
}

export function concatenateNames(items: Array<{name: string}>): string {
  let result = '';
  for (let i = 0; i < items.length; i++) {
    result += items[i].name;
  }
  return result;
}