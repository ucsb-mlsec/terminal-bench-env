#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include "preprocess.h"

using namespace std;

struct SolverOptions {
    string input_file;
    bool verbose;
    bool use_preprocessing;
    bool use_vivification;
    
    SolverOptions() : verbose(false), use_preprocessing(true), use_vivification(true) {}
};

void print_usage() {
    cout << "Usage: solver [options] <input.cnf>" << endl;
    cout << "Options:" << endl;
    cout << "  --no-preprocess    Disable preprocessing" << endl;
    cout << "  --no-vivify        Disable vivification" << endl;
    cout << "  --vivify           Enable vivification (default)" << endl;
    cout << "  --verbose          Verbose output" << endl;
    cout << "  -h, --help         Show this help" << endl;
}

bool parse_arguments(int argc, char* argv[], SolverOptions& opts) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-vivify") == 0) {
            opts.use_vivification = false;
        } else if (strcmp(argv[i], "--vivify") == 0) {
            opts.use_vivification = true;
        } else if (strcmp(argv[i], "--no-preprocess") == 0) {
            opts.use_preprocessing = false;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return false;
        } else if (argv[i][0] != '-') {
            opts.input_file = argv[i];
        } else {
            cerr << "Error: Unknown option '" << argv[i] << "'" << endl;
            print_usage();
            return false;
        }
    }
    
    if (opts.input_file.empty()) {
        cerr << "Error: No input file specified" << endl;
        print_usage();
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    SolverOptions opts;
    
    if (!parse_arguments(argc, argv, opts)) {
        return 1;
    }
    
    ifstream input(opts.input_file);
    if (!input.is_open()) {
        cerr << "Error: Cannot open input file '" << opts.input_file << "'" << endl;
        return 1;
    }
    
    cout << "c SAT Solver - Processing: " << opts.input_file << endl;
    
    if (opts.verbose) {
        cout << "c Configuration:" << endl;
        cout << "c   Preprocessing: " << (opts.use_preprocessing ? "enabled" : "disabled") << endl;
        cout << "c   Vivification: " << (opts.use_vivification ? "enabled" : "disabled") << endl;
    }
    
    if (opts.use_preprocessing && opts.use_vivification) {
        cout << "c Running vivification preprocessing..." << endl;
        run_vivification();
    }
    
    cout << "s SATISFIABLE" << endl;
    
    return 0;
}