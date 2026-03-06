/*
 * Producer-Consumer System in Promela
 * 
 * LTL Properties to verify:
 * 1. Safety: "The buffer never overflows"
 *    [] (buffer_count <= MAX_SIZE)
 * 
 * 2. Liveness: "Items produced are eventually consumed"
 *    [] (item_produced -> <> item_consumed)
 * 
 * 3. Mutual Exclusion: "Producer and consumer never access buffer simultaneously"
 *    [] !(producer_active && consumer_active)
 */

#define MAX_SIZE 5

/* Global variables */
byte buffer_count = 0;
bool item_produced = false;
bool item_consumed = false;
bool producer_active = false;
bool consumer_active = false;

/* Producer process */
proctype producer() {
    do
    :: (buffer_count < MAX_SIZE) ->
        /* Enter critical section */
        producer_active = true;
        
        /* Produce an item */
        buffer_count++;
        item_produced = true;
        
        /* Exit critical section */
        producer_active = false;
        
        /* Reset flag after a step */
        item_produced = false;
    od;
}

/* Consumer process */
proctype consumer() {
    do
    :: (buffer_count > 0) ->
        /* Enter critical section */
        consumer_active = true;
        
        /* Consume an item */
        buffer_count--;
        item_consumed = true;
        
        /* Exit critical section */
        consumer_active = false;
        
        /* Reset flag after a step */
        item_consumed = false;
    od;
}

/* Initialization */
init {
    atomic {
        run producer();
        run consumer();
    }
}