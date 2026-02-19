#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Physical constants */
#define COULOMB_CONSTANT 332.0636  /* kcal*A/(mol*e^2) */
#define MAX_ATOMS 1000
#define MAX_LINE 256

/* Structure to hold atomic data */
typedef struct {
    double x, y, z;      /* Cartesian coordinates in Angstroms */
    double charge;       /* Partial charge in elementary charge units */
    double epsilon;      /* LJ well depth in kcal/mol */
    double sigma;        /* LJ distance parameter in Angstroms */
} Atom;

/**
 * Calculate Lennard-Jones potential energy between two atoms
 * 
 * The Lennard-Jones potential models van der Waals interactions:
 * V(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
 * 
 * Parameters:
 *   r       - distance between atoms (Angstroms)
 *   epsilon - depth of potential well (kcal/mol)
 *   sigma   - distance at which potential is zero (Angstroms)
 * 
 * Returns: potential energy in kcal/mol
 */
double calculate_lennard_jones(double r, double epsilon, double sigma) {
    if (r < 0.0001) return 0.0;  /* Avoid division by zero */
    
    double sr = sigma / r;
    double sr6 = sr * sr * sr * sr * sr * sr;
    double sr12 = sr6 * sr6;
    
    return 4.0 * epsilon * (sr12 - sr6);
}

/**
 * Calculate Coulomb electrostatic energy between two charged particles
 * 
 * Coulomb's law for electrostatic interactions:
 * V(r) = (q1*q2) / r
 * 
 * Parameters:
 *   q1 - charge on first atom (elementary charge units)
 *   q2 - charge on second atom (elementary charge units)
 *   r  - distance between atoms (Angstroms)
 * 
 * Returns: electrostatic energy in kcal/mol
 */
double calculate_coulomb(double q1, double q2, double r) {
    if (r < 0.0001) return 0.0;  /* Avoid division by zero */
    
    return COULOMB_CONSTANT * (q1 * q2) / r;
}

/**
 * Calculate distance between two atoms
 */
static double calculate_distance(const Atom *a1, const Atom *a2) {
    double dx = a1->x - a2->x;
    double dy = a1->y - a2->y;
    double dz = a1->z - a2->z;
    
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/**
 * Compute total molecular energy from configuration file
 * 
 * This is the main exported function that drives the energy calculation.
 * It reads atomic coordinates and parameters from a configuration file,
 * computes all pairwise interactions, and writes results to output.
 * 
 * Config file format (per line):
 *   x y z charge epsilon sigma
 * 
 * Parameters:
 *   config_file - path to input configuration file
 *   output_file - path to output results file
 * 
 * Returns: 0 on success, -1 on error
 */
int compute_molecular_energy(const char* config_file, const char* output_file) {
    FILE *fin = NULL;
    FILE *fout = NULL;
    Atom atoms[MAX_ATOMS];
    int num_atoms = 0;
    char line[MAX_LINE];
    double total_energy = 0.0;
    double lj_energy = 0.0;
    double coulomb_energy = 0.0;
    
    /* Open configuration file */
    fin = fopen(config_file, "r");
    if (!fin) {
        fprintf(stderr, "Error: Cannot open configuration file %s\n", config_file);
        return -1;
    }
    
    /* Read atomic data */
    while (fgets(line, MAX_LINE, fin) && num_atoms < MAX_ATOMS) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n') continue;
        
        int ret = sscanf(line, "%lf %lf %lf %lf %lf %lf",
                        &atoms[num_atoms].x,
                        &atoms[num_atoms].y,
                        &atoms[num_atoms].z,
                        &atoms[num_atoms].charge,
                        &atoms[num_atoms].epsilon,
                        &atoms[num_atoms].sigma);
        
        if (ret == 6) {
            num_atoms++;
        }
    }
    fclose(fin);
    
    if (num_atoms == 0) {
        fprintf(stderr, "Error: No atoms read from configuration file\n");
        return -1;
    }
    
    /* Calculate pairwise interactions */
    for (int i = 0; i < num_atoms; i++) {
        for (int j = i + 1; j < num_atoms; j++) {
            double r = calculate_distance(&atoms[i], &atoms[j]);
            
            /* Combine LJ parameters using Lorentz-Berthelot rules */
            double epsilon_ij = sqrt(atoms[i].epsilon * atoms[j].epsilon);
            double sigma_ij = (atoms[i].sigma + atoms[j].sigma) / 2.0;
            
            /* Calculate Lennard-Jones contribution */
            double lj = calculate_lennard_jones(r, epsilon_ij, sigma_ij);
            lj_energy += lj;
            
            /* Calculate Coulomb contribution */
            double coul = calculate_coulomb(atoms[i].charge, atoms[j].charge, r);
            coulomb_energy += coul;
        }
    }
    
    total_energy = lj_energy + coulomb_energy;
    
    /* Write results to output file */
    fout = fopen(output_file, "w");
    if (!fout) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
        return -1;
    }
    
    fprintf(fout, "%.2f\n", total_energy);
    fprintf(fout, "TOTAL_ENERGY: %.2f kcal/mol\n", total_energy);
    fprintf(fout, "LJ_ENERGY: %.2f kcal/mol\n", lj_energy);
    fprintf(fout, "COULOMB_ENERGY: %.2f kcal/mol\n", coulomb_energy);
    fprintf(fout, "NUM_ATOMS: %d\n", num_atoms);
    
    fclose(fout);
    
    return 0;
}