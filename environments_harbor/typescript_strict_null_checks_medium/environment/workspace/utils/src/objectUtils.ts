export function getProperty(obj: any, key: string): any {
  return obj[key];
}

export function merge<T>(target: T, source: Partial<T>): T {
  return Object.assign(target, source);
}

export function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

export function getNestedValue(obj: any, path: string): any {
  const parts = path.split('.');
  let result = obj;
  for (const part of parts) {
    result = result[part];
  }
  return result;
}