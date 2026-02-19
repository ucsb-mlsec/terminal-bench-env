#!/usr/bin/env python3

import time
import tracemalloc
import json
import sys
from entity_manager import EntityManager

def load_entity_types():
    """Loads entity configuration from 'entity_types.json'"""
    try:
        with open('entity_types.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: entity_types.json not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in entity_types.json")
        sys.exit(1)

def simulate_gameplay(manager, entity_types):
    """
    Simulates 60 seconds of gameplay at 60 FPS (3600 frames).
    Generates approximately 50,000 lifecycle events.
    """
    FPS = 60
    DURATION_SECONDS = 60
    TOTAL_FRAMES = FPS * DURATION_SECONDS
    
    # Spawn rates per second
    ENEMY_SPAWN_RATE = 5
    PROJECTILE_SPAWN_RATE = 10
    PARTICLE_SPAWN_RATE = 15
    POWERUP_SPAWN_RATE = 1
    
    # Lifetimes in frames
    PROJECTILE_LIFETIME = 2 * FPS  # 2 seconds
    PARTICLE_LIFETIME = 1 * FPS    # 1 second
    ENEMY_LIFETIME = 5 * FPS       # 5 seconds
    POWERUP_LIFETIME = 10 * FPS    # 10 seconds
    
    # Track spawned entities
    active_entities = {
        'enemy': {},
        'projectile': {},
        'particle': {},
        'powerup': {}
    }
    
    next_entity_id = 0
    frame_counter = 0
    
    for frame in range(TOTAL_FRAMES):
        frame_counter += 1
        
        # Spawn enemies
        if frame % (FPS // ENEMY_SPAWN_RATE) == 0:
            entity_id = next_entity_id
            next_entity_id += 1
            entity_data = {
                'type': 'enemy',
                'position': [frame % 100, (frame * 2) % 100],
                'velocity': [1.0, 0.5],
                'health': 100
            }
            manager.spawn_entity(entity_id, entity_data)
            active_entities['enemy'][entity_id] = frame
        
        # Spawn projectiles
        if frame % (FPS // PROJECTILE_SPAWN_RATE) == 0:
            entity_id = next_entity_id
            next_entity_id += 1
            entity_data = {
                'type': 'projectile',
                'position': [frame % 150, (frame * 3) % 150],
                'velocity': [5.0, 0.0],
                'damage': 25
            }
            manager.spawn_entity(entity_id, entity_data)
            active_entities['projectile'][entity_id] = frame
        
        # Spawn particles
        if frame % (FPS // PARTICLE_SPAWN_RATE) == 0:
            entity_id = next_entity_id
            next_entity_id += 1
            entity_data = {
                'type': 'particle',
                'position': [frame % 200, (frame * 4) % 200],
                'velocity': [2.0, -1.0],
                'color': [255, 128, 0]
            }
            manager.spawn_entity(entity_id, entity_data)
            active_entities['particle'][entity_id] = frame
        
        # Spawn powerups
        if frame % (FPS // POWERUP_SPAWN_RATE) == 0:
            entity_id = next_entity_id
            next_entity_id += 1
            entity_data = {
                'type': 'powerup',
                'position': [frame % 80, (frame * 5) % 80],
                'velocity': [0.0, 0.0],
                'bonus': 50
            }
            manager.spawn_entity(entity_id, entity_data)
            active_entities['powerup'][entity_id] = frame
        
        # Update entity positions
        for entity_type in ['enemy', 'projectile', 'particle', 'powerup']:
            for entity_id in list(active_entities[entity_type].keys()):
                entity = manager.get_entity(entity_id)
                if entity:
                    entity['position'][0] += entity['velocity'][0]
                    entity['position'][1] += entity['velocity'][1]
                    manager.update_entity(entity_id, entity)
        
        # Despawn projectiles after lifetime
        for entity_id, spawn_frame in list(active_entities['projectile'].items()):
            if frame - spawn_frame >= PROJECTILE_LIFETIME:
                manager.despawn_entity(entity_id)
                del active_entities['projectile'][entity_id]
        
        # Despawn particles after lifetime
        for entity_id, spawn_frame in list(active_entities['particle'].items()):
            if frame - spawn_frame >= PARTICLE_LIFETIME:
                manager.despawn_entity(entity_id)
                del active_entities['particle'][entity_id]
        
        # Despawn enemies after lifetime
        for entity_id, spawn_frame in list(active_entities['enemy'].items()):
            if frame - spawn_frame >= ENEMY_LIFETIME:
                manager.despawn_entity(entity_id)
                del active_entities['enemy'][entity_id]
        
        # Despawn powerups after lifetime
        for entity_id, spawn_frame in list(active_entities['powerup'].items()):
            if frame - spawn_frame >= POWERUP_LIFETIME:
                manager.despawn_entity(entity_id)
                del active_entities['powerup'][entity_id]

def main():
    """Main benchmark execution"""
    # Start memory tracking
    tracemalloc.start()
    
    # Load entity types
    entity_types = load_entity_types()
    
    # Create entity manager
    manager = EntityManager()
    
    # Record start time
    start_time = time.time()
    
    # Run simulation
    simulate_gameplay(manager, entity_types)
    
    # Record end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Get peak memory usage in MB
    current, peak = tracemalloc.get_traced_memory()
    peak_memory_mb = peak / (1024 * 1024)
    
    # Get checksum from manager
    checksum = manager.get_checksum()
    
    # Print results
    print("Performance Benchmark Results:")
    print("============================")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Peak Memory: {peak_memory_mb:.1f} MB")
    print(f"Verification Checksum: {checksum}")
    print()
    print("Benchmark Complete!")
    
    # Stop memory tracking
    tracemalloc.stop()

if __name__ == '__main__':
    main()