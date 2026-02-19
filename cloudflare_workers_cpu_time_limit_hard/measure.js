import { processWebhook as original } from './processor.js';
import fs from 'fs';

let solution;
try {
  const solutionModule = await import('./solution.js');
  solution = solutionModule.processWebhook;
} catch (e) {
  console.log('⚠️  Solution file not found at ./solution.js');
  console.log('Please create your optimized implementation first.\n');
  process.exit(1);
}

function measureCPU(fn, payload) {
  const startUsage = process.cpuUsage();
  const result = fn(payload);
  const endUsage = process.cpuUsage(startUsage);
  const cpuTimeMs = (endUsage.user + endUsage.system) / 1000;
  return { result, cpuTimeMs };
}

function compareOutputs(obj1, obj2) {
  return JSON.stringify(obj1, Object.keys(obj1).sort()) === JSON.stringify(obj2, Object.keys(obj2).sort());
}

console.log('🔍 Cloudflare Worker Performance Test\n');
console.log('=' .repeat(70));

const payloads = [
  { name: 'small', path: './payloads/small.json' },
  { name: 'medium', path: './payloads/medium.json' },
  { name: 'large', path: './payloads/large.json' }
];

let allPassed = true;
const results = [];

for (const { name, path } of payloads) {
  try {
    const payload = JSON.parse(fs.readFileSync(path, 'utf8'));
    
    // Warm up
    original(payload);
    solution(payload);
    
    // Measure original
    const originalMeasure = measureCPU(original, payload);
    
    // Measure solution
    const solutionMeasure = measureCPU(solution, payload);
    
    // Compare outputs
    const correct = compareOutputs(originalMeasure.result, solutionMeasure.result);
    
    const improvement = ((originalMeasure.cpuTimeMs - solutionMeasure.cpuTimeMs) / originalMeasure.cpuTimeMs * 100).toFixed(1);
    
    results.push({
      name,
      correct,
      originalTime: originalMeasure.cpuTimeMs,
      solutionTime: solutionMeasure.cpuTimeMs,
      improvement
    });
    
    if (!correct) {
      allPassed = false;
    }
    
  } catch (e) {
    console.log(`❌ Error loading ${name} payload: ${e.message}\n`);
    allPassed = false;
  }
}

// Display results
console.log('\n📊 Test Results:\n');
console.log('Payload'.padEnd(12) + 'Correctness'.padEnd(15) + 'Original'.padEnd(15) + 'Solution'.padEnd(15) + 'Improvement');
console.log('-'.repeat(70));

for (const r of results) {
  const correctStr = r.correct ? '✅ PASS' : '❌ FAIL';
  const improvementStr = r.improvement > 0 ? `+${r.improvement}%` : `${r.improvement}%`;
  
  console.log(
    r.name.padEnd(12) +
    correctStr.padEnd(15) +
    `${r.originalTime.toFixed(2)}ms`.padEnd(15) +
    `${r.solutionTime.toFixed(2)}ms`.padEnd(15) +
    improvementStr
  );
}

console.log('='.repeat(70));

// Overall result
const hasImprovement = results.some(r => r.solutionTime < r.originalTime);
const allCorrect = results.every(r => r.correct);

if (allCorrect && hasImprovement) {
  console.log('\n✅ SUCCESS: All tests passed with performance improvements!\n');
} else if (!allCorrect) {
  console.log('\n❌ FAILURE: Output correctness issues detected.\n');
  console.log('Your solution must produce identical results to the original.\n');
} else {
  console.log('\n⚠️  WARNING: Output is correct but no performance improvement detected.\n');
}