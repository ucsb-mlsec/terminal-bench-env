#ifndef BUILD_CONFIG_H
#define BUILD_CONFIG_H

// Build configuration - Auto-generated
// Do not edit manually

#define SOLVER_VERSION "1.0.0"
#define BUILD_DATE "2024-01-15"

// Feature flags
#define FEATURE_PREPROCESSING 1
#define FEATURE_SUBSUMPTION 1
#define FEATURE_VARIABLE_ELIMINATION 0

// Vivification configuration
// Comment out to disable vivification at compile time
#define FEATURE_VIVIFICATION 1

#ifdef FEATURE_VIVIFICATION
  #define VIVIFY_DEFAULT_ENABLED 1
  #define VIVIFY_MAX_EFFORT 3
#else
  #define VIVIFY_DEFAULT_ENABLED 0
#endif

// Performance settings
#define DEFAULT_RESTART_INTERVAL 100
#define MAX_LEARNED_CLAUSES 10000

// Debug options
#ifdef DEBUG
  #define ENABLE_ASSERTIONS 1
  #define VERBOSE_PREPROCESSING 1
#else
  #define ENABLE_ASSERTIONS 0
  #define VERBOSE_PREPROCESSING 0
#endif

// Clause database settings
#define CLAUSE_INITIAL_CAPACITY 1024
#define CLAUSE_GROWTH_FACTOR 2

// Preprocessing limits
#define MAX_PREPROCESS_ITERATIONS 5
#define SUBSUMPTION_LIMIT 1000

// Memory management
#define ENABLE_MEMORY_POOLING 1
#define POOL_BLOCK_SIZE 4096

#endif // BUILD_CONFIG_H