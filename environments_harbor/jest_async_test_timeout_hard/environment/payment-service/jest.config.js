module.exports = {
  testEnvironment: 'node',
  testTimeout: 10000,
  coveragePathIgnorePatterns: [
    '/node_modules/'
  ],
  testMatch: [
    '**/__tests__/**/*.js',
    '**/?(*.)+(spec|test).js'
  ],
  collectCoverageFrom: [
    '**/*.js',
    '!**/node_modules/**',
    '!**/coverage/**'
  ],
  verbose: true
};