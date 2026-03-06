#!/usr/bin/env python3

import random

def simulate_race():
    # Fixed random seed for deterministic results
    random.seed(42)
    
    # Define competitors
    racer_a = {
        'name': 'Racer_A',
        'speed': 10,
        'position': 0
    }
    
    racer_b = {
        'name': 'Racer_B',
        'speed': 8,
        'position': 0
    }
    
    race_distance = 100
    iteration = 0
    
    print("=== Race Simulation Started ===")
    print(f"Race Distance: {race_distance} units")
    print(f"{racer_a['name']} - Initial Speed: {racer_a['speed']}")
    print(f"{racer_b['name']} - Initial Speed: {racer_b['speed']}")
    print()
    
    # Run the race
    while racer_a['position'] < race_distance and racer_b['position'] < race_distance:
        iteration += 1
        
        # Apply speed variations (fatigue/boost effects)
        speed_variation_a = random.uniform(-0.5, 1.0)
        speed_variation_b = random.uniform(-0.5, 1.0)
        
        # Update positions
        racer_a['position'] += racer_a['speed'] + speed_variation_a
        racer_b['position'] += racer_b['speed'] + speed_variation_b
        
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: {racer_a['name']} at {racer_a['position']:.2f}, {racer_b['name']} at {racer_b['position']:.2f}")
    
    print()
    print("=== Race Finished ===")
    print(f"Final Position - {racer_a['name']}: {racer_a['position']:.2f}")
    print(f"Final Position - {racer_b['name']}: {racer_b['position']:.2f}")
    print()
    
    # BUG: Declaring winner based on who has LOWER position instead of HIGHER
    # The correct logic should be: whoever has the higher position crossed finish line first
    # But the code checks for lower position
    if racer_a['position'] < racer_b['position']:
        winner = racer_a['name']
    else:
        winner = racer_b['name']
    
    print(f"🏆 Winner: {winner}")
    print()
    
    return winner

if __name__ == "__main__":
    simulate_race()