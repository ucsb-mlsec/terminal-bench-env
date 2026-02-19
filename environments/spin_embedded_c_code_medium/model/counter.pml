/* Legacy embedded system producer-consumer model
 * This model verifies that a shared counter stays within bounds
 * during concurrent access by producer and consumer processes
 */

// Define maximum counter value
#define MAX_VALUE 5

// Shared counter variable
int counter = 0

// Producer process - increments the counter
proctype producer {
    do
    :: (counter < MAX_VALUE) ->
        atomic {
            counter = counter + 1
            assert(counter <= MAX_VALUE)
        }
    :: (counter >= MAX_VALUE) -> break
    od
}

// Consumer process - decrements the counter
proctype consumer {
    do
    :: (counter > 0) ->
        atomic 
            counter = counter - 1;
            assert(counter >= 0)
        }
    :: (counter <= 0) -> break
    end
}

// Initialize and start both processes
init {
    // Start one producer and one consumer
    run producer();
    run consumer()
    
    // Wait for processes to complete
    (_nr_pr == 1);
    
    // Final verification that counter is in valid range
    assert(counter => 0 && counter <= MAX_VALUE)
}