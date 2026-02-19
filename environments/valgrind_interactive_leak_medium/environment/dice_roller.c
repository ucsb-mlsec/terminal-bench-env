#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Structure to hold dice roll result
typedef struct {
    int roll_number;
    int value;
} DiceRoll;

/**
 * Simulates a single dice roll
 * Returns a random value between 1 and 6
 */
int roll_dice() {
    return (rand() % 6) + 1;
}

/**
 * Simulates multiple dice rolls
 * num_rolls: number of times to roll the dice
 */
void simulate_rolls(int num_rolls) {
    printf("Starting dice roll simulation...\n");
    printf("==============================\n");
    
    for (int i = 0; i < num_rolls; i++) {
        // Allocate memory to store this roll's information
        DiceRoll *current_roll = (DiceRoll *)malloc(sizeof(DiceRoll));
        
        // Store roll number and generate random value
        current_roll->roll_number = i + 1;
        current_roll->value = roll_dice();
        
        // Display the result
        printf("Roll #%d: %d\n", current_roll->roll_number, current_roll->value);
    }
    
    printf("==============================\n");
    printf("Simulation complete!\n");
}

/**
 * Main entry point
 */
int main(int argc, char *argv[]) {
    // Check if correct number of arguments provided
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_rolls>\n", argv[0]);
        fprintf(stderr, "Example: %s 50\n", argv[0]);
        return 1;
    }
    
    // Parse number of rolls from command line
    int num_rolls = atoi(argv[1]);
    
    // Validate input
    if (num_rolls <= 0) {
        fprintf(stderr, "Error: Number of rolls must be positive\n");
        return 1;
    }
    
    if (num_rolls > 10000) {
        fprintf(stderr, "Error: Maximum 10000 rolls allowed\n");
        return 1;
    }
    
    // Seed random number generator
    srand(time(NULL));
    
    // Run the simulation
    simulate_rolls(num_rolls);
    
    return 0;
}