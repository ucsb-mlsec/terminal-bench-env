#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROOM_NAME 64
#define MAX_TREASURE_DESC 128

typedef struct {
    char name[MAX_ROOM_NAME];
    int danger_level;
    int treasure_count;
} Room;

typedef struct {
    char *description;
    int value;
} Treasure;

// Global game state
int total_treasures_found = 0;

// Function prototypes
Room* create_room(int room_number);
char* generate_treasure(int room_number);
void explore_room(int room_number);
void cleanup_room(Room *room);
void print_stats();

// Creates a room structure
Room* create_room(int room_number) {
    Room *room = (Room*)malloc(sizeof(Room));
    if (!room) {
        fprintf(stderr, "Failed to allocate room\n");
        exit(1);
    }
    
    snprintf(room->name, MAX_ROOM_NAME, "Chamber_%d", room_number);
    room->danger_level = (room_number % 5) + 1;
    room->treasure_count = (room_number % 3) + 1;
    
    return room;
}

// Generates treasure descriptions - THIS FUNCTION HAS A MEMORY LEAK
char* generate_treasure(int room_number) {
    char *treasure_desc = (char*)malloc(MAX_TREASURE_DESC);
    if (!treasure_desc) {
        fprintf(stderr, "Failed to allocate treasure description\n");
        exit(1);
    }
    
    // Generate different treasure types based on room number
    int treasure_type = room_number % 5;
    
    switch(treasure_type) {
        case 0:
            snprintf(treasure_desc, MAX_TREASURE_DESC, 
                    "Golden coins scattered on the floor (Room %d)", room_number);
            break;
        case 1:
            snprintf(treasure_desc, MAX_TREASURE_DESC,
                    "Ancient sword mounted on the wall (Room %d)", room_number);
            break;
        case 2:
            snprintf(treasure_desc, MAX_TREASURE_DESC,
                    "Mysterious glowing gem on a pedestal (Room %d)", room_number);
            break;
        case 3:
            snprintf(treasure_desc, MAX_TREASURE_DESC,
                    "Chest full of precious jewels (Room %d)", room_number);
            break;
        case 4:
            snprintf(treasure_desc, MAX_TREASURE_DESC,
                    "Ornate crown with embedded rubies (Room %d)", room_number);
            break;
    }
    
    total_treasures_found++;
    
    // MEMORY LEAK: treasure_desc is never freed!
    // The caller doesn't know they need to free it
    return treasure_desc;
}

// Explores a single room
void explore_room(int room_number) {
    printf("Exploring room %d...\n", room_number);
    
    // Create the room
    Room *current_room = create_room(room_number);
    
    // Generate treasures for this room
    for (int i = 0; i < current_room->treasure_count; i++) {
        char *treasure = generate_treasure(room_number);
        // We get the treasure description but never store or free it
        // This is where the leak occurs - generate_treasure allocates
        // but we don't free the returned pointer
    }
    
    // Simulate some room exploration activity
    if (current_room->danger_level > 3) {
        printf("  Warning: High danger level in %s!\n", current_room->name);
    }
    
    // Properly cleanup the room structure
    cleanup_room(current_room);
}

// Properly cleans up a room
void cleanup_room(Room *room) {
    if (room) {
        free(room);
    }
}

// Prints game statistics
void print_stats() {
    printf("\n=== Game Statistics ===\n");
    printf("Total treasures found: %d\n", total_treasures_found);
}

// Helper function to allocate temporary data for room atmosphere
char* generate_atmosphere(int room_number) {
    char *atmosphere = (char*)malloc(100);
    if (!atmosphere) {
        return NULL;
    }
    
    snprintf(atmosphere, 100, "Room %d has a mysterious atmosphere", room_number);
    return atmosphere;
}

// Main game loop
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_rooms>\n", argv[0]);
        return 1;
    }
    
    int num_rooms = atoi(argv[1]);
    
    if (num_rooms <= 0) {
        fprintf(stderr, "Number of rooms must be positive\n");
        return 1;
    }
    
    printf("Starting dungeon exploration...\n");
    printf("Planning to explore %d rooms\n\n", num_rooms);
    
    // Main exploration loop
    for (int i = 1; i <= num_rooms; i++) {
        explore_room(i);
        
        // Generate and properly cleanup atmosphere descriptions
        char *atm = generate_atmosphere(i);
        if (atm) {
            free(atm);  // This is properly freed
        }
    }
    
    printf("\nExploration complete!\n");
    print_stats();
    
    return 0;
}