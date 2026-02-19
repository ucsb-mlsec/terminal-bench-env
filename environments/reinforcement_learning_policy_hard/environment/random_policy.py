#!/usr/bin/env python3

import random
import json
import os

class WarehouseEnv:
    """
    Warehouse environment for robot navigation.
    10x10 grid with obstacles, robot moves in 4 directions.
    """
    
    def __init__(self, warehouse_layout):
        """
        Initialize the warehouse environment.
        
        Args:
            warehouse_layout: List of strings representing the 10x10 grid
                            '.' = open space, '#' = obstacle
        """
        self.layout = warehouse_layout
        self.grid_size = 10
        self.max_steps = 100
        
        # Parse obstacles from layout
        self.obstacles = set()
        for y in range(len(warehouse_layout)):
            for x in range(len(warehouse_layout[y])):
                if warehouse_layout[y][x] == '#':
                    self.obstacles.add((x, y))
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.action_effects = {
            0: (0, -1),  # up (decrease y)
            1: (0, 1),   # down (increase y)
            2: (-1, 0),  # left (decrease x)
            3: (1, 0)    # right (increase x)
        }
        
        self.current_position = None
        self.goal_position = None
        self.steps_taken = 0
    
    def reset(self, start, goal):
        """
        Reset environment with new start and goal positions.
        
        Args:
            start: [x, y] starting position
            goal: [gx, gy] goal position
            
        Returns:
            state: Dictionary with 'position' and 'goal' keys
        """
        self.current_position = list(start)
        self.goal_position = list(goal)
        self.steps_taken = 0
        
        return {
            'position': list(self.current_position),
            'goal': list(self.goal_position)
        }
    
    def step(self, action):
        """
        Execute action and return next state, reward, and done flag.
        
        Args:
            action: Integer 0-3 representing direction
            
        Returns:
            next_state: Dictionary with 'position' and 'goal'
            reward: Float reward value
            done: Boolean indicating episode termination
        """
        self.steps_taken += 1
        
        # Calculate new position
        dx, dy = self.action_effects[action]
        new_x = self.current_position[0] + dx
        new_y = self.current_position[1] + dy
        
        # Check for boundaries
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            # Hit boundary - penalty and stay in place
            reward = -10
            done = False
        # Check for obstacles
        elif (new_x, new_y) in self.obstacles:
            # Hit obstacle - penalty and stay in place
            reward = -10
            done = False
        else:
            # Valid move
            self.current_position = [new_x, new_y]
            
            # Check if goal reached
            if self.current_position == self.goal_position:
                reward = 10
                done = True
            else:
                # Step penalty to encourage shorter paths
                reward = -1
                done = False
        
        # Check max steps
        if self.steps_taken >= self.max_steps:
            done = True
        
        next_state = {
            'position': list(self.current_position),
            'goal': list(self.goal_position)
        }
        
        return next_state, reward, done


class RandomPolicy:
    """
    Baseline random policy that selects actions uniformly at random.
    """
    
    def get_action(self, state):
        """
        Select a random action regardless of state.
        
        Args:
            state: Dictionary with 'position' and 'goal' keys
            
        Returns:
            action: Random integer 0-3
        """
        return random.choice([0, 1, 2, 3])


def load_warehouse_layout(filepath):
    """Load warehouse layout from file."""
    with open(filepath, 'r') as f:
        layout = [line.strip() for line in f.readlines()]
    return layout


def test_policy(env, policy, test_scenarios):
    """Test policy on given scenarios and return average reward."""
    total_reward = 0
    
    for scenario in test_scenarios:
        start = scenario['start']
        goal = scenario['goal']
        
        state = env.reset(start, goal)
        episode_reward = 0
        done = False
        
        while not done:
            action = policy.get_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    average_reward = total_reward / len(test_scenarios)
    return average_reward


if __name__ == "__main__":
    # Load warehouse layout
    layout = load_warehouse_layout('/workspace/warehouse_layout.txt')
    
    # Create environment and random policy
    env = WarehouseEnv(layout)
    policy = RandomPolicy()
    
    # Test on a few example scenarios
    example_scenarios = [
        {'start': [0, 0], 'goal': [9, 9]},
        {'start': [5, 5], 'goal': [2, 8]},
        {'start': [0, 9], 'goal': [9, 0]},
        {'start': [3, 3], 'goal': [7, 7]},
        {'start': [1, 1], 'goal': [8, 8]}
    ]
    
    print("Testing Random Policy Baseline...")
    print(f"Number of test scenarios: {len(example_scenarios)}")
    
    avg_reward = test_policy(env, policy, example_scenarios)
    
    print(f"Average reward: {avg_reward:.2f}")
    print("(Expected range: -15 to -20 for random policy)")