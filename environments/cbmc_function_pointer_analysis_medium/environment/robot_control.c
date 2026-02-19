#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NAVIGATE 0
#define GRIPPER 1
#define COLLISION 2

// Function pointer type for operation callbacks
typedef void (*operation_callback)(int);

// Callback function for navigation operations
void navigate_callback(int param) {
    printf("Navigating to position: %d\n", param);
}

// Callback function for gripper control
void gripper_callback(int param) {
    printf("Gripper action: %d\n", param);
}

// Callback function for collision detection
void collision_callback(int param) {
    printf("Collision detected at sensor: %d\n", param);
}

// Global array of function pointers - dynamically initialized
operation_callback callback_table[3];

// Runtime initialization function that prevents static resolution
void initialize_callbacks(void) {
    // Dynamic assignment at runtime prevents CBMC from statically
    // determining which functions are in the callback table
    int nav_op = NAVIGATE;
    int grip_op = GRIPPER;
    int coll_op = COLLISION;
    
    // Use variables to index the array, making it harder for
    // static analysis to resolve
    callback_table[nav_op] = navigate_callback;
    callback_table[grip_op] = gripper_callback;
    callback_table[coll_op] = collision_callback;
}

// Dispatch function that calls appropriate callback based on operation type
void dispatch(int operation_type, int param) {
    // Bounds checking for safety
    if (operation_type >= 0 && operation_type < 3) {
        // Indirect function call through pointer array
        // CBMC cannot statically resolve which function will be called
        callback_table[operation_type](param);
    } else {
        printf("Error: Invalid operation type %d\n", operation_type);
    }
}

int main(void) {
    // Initialize the callback system at runtime
    initialize_callbacks();
    
    // Execute various robot operations
    dispatch(NAVIGATE, 42);
    dispatch(GRIPPER, 1);
    dispatch(COLLISION, 7);
    dispatch(NAVIGATE, 15);
    dispatch(GRIPPER, 0);
    
    return 0;
}