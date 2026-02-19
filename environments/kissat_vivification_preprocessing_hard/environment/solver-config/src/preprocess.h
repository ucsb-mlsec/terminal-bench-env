#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <vector>
#include <string>

// Preprocessing configuration for SAT solver
// This class manages various preprocessing techniques including
// vivification, subsumption, and other clause simplification methods

class PreprocessConfig {
public:
    bool enable_preprocessing;
    bool enable_vivification;
    bool enable_subsumption;
    bool enable_variable_elimination;
    int vivification_effort;
    int max_vivification_rounds;
    
    PreprocessConfig() {
        enable_preprocessing = true;
        enable_subsumption = true;
        enable_variable_elimination = true;
        vivification_effort = 2;
        max_vivification_rounds = 3;
        
        // Vivification is controlled at compile time
#ifdef ENABLE_VIVIFICATION
        enable_vivification = true;
#else
        enable_vivification = false;
#endif
    }
    
    void print_config() const;
    bool is_vivification_enabled() const {
        return enable_vivification;
    }
};

// Vivification-specific functionality
#ifdef ENABLE_VIVIFICATION
#define VIVIFICATION_ACTIVE 1

// Run vivification preprocessing on clause database
void run_vivification(std::vector<std::vector<int>>& clauses, int effort);

// Check if a clause can be simplified through vivification
bool try_vivify_clause(const std::vector<int>& clause);

#else
#define VIVIFICATION_ACTIVE 0
#endif

// Other preprocessing functions
void run_subsumption(std::vector<std::vector<int>>& clauses);
void eliminate_variables(std::vector<std::vector<int>>& clauses);

#endif // PREPROCESS_H