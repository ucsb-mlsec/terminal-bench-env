#!/usr/bin/env python3

import numpy as np
import random
from gridworld import GridWorld

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Hyperparameters
learning_rate = 0.1
discount = 0.95
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
num_training_episodes = 5000
max_steps_per_episode = 100

def initialize_q_table():
    """Initialize Q-table as nested dictionary with all state-action pairs set to 0.0"""
    q_table = {}
    for row in range(5):
        for col in range(5):
            state = (row, col)
            q_table[state] = {}
            for action in range(4):  # 4 actions: up, down, left, right
                q_table[state][action] = 0.0
    return q_table

def epsilon_greedy(state, epsilon, q_table):
    """Select action using epsilon-greedy policy"""
    if random.random() < epsilon:
        # Explore: choose random action
        return random.randint(0, 3)
    else:
        # Exploit: choose action with highest Q-value
        q_values = q_table[state]
        max_q = max(q_values.values())
        # Handle ties by randomly selecting among actions with max Q-value
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

def get_max_q_value(state, q_table):
    """Get maximum Q-value for a given state"""
    return max(q_table[state].values())

def train(env, q_table):
    """Train the agent using Q-learning"""
    epsilon = epsilon_start
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_training_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps_per_episode:
            # Select action using epsilon-greedy policy
            action = epsilon_greedy(state, epsilon, q_table)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Q-learning update
            # Q(s,a) = Q(s,a) + learning_rate * (reward + discount * max(Q(s',a')) - Q(s,a))
            current_q = q_table[state][action]
            
            if done:
                # Terminal state, no future rewards
                target_q = reward
            else:
                # Non-terminal state
                max_next_q = get_max_q_value(next_state, q_table)
                target_q = reward + discount * max_next_q
            
            # Update Q-value
            q_table[state][action] = current_q + learning_rate * (target_q - current_q)
            
            # Update state and tracking variables
            state = next_state
            total_reward += reward
            steps += 1
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Track episode statistics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Print progress every 500 episodes
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode + 1}/{num_training_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Avg Steps (last 100): {avg_steps:.2f}, "
                  f"Epsilon: {epsilon:.4f}")
    
    return q_table

def evaluate(env, q_table, num_episodes=100):
    """Evaluate the learned policy"""
    total_steps = 0
    successes = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        steps = 0
        done = False
        
        while not done and steps < max_steps_per_episode:
            # Use greedy policy (epsilon = 0)
            action = epsilon_greedy(state, 0.0, q_table)
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1
            
            if done and reward > 0:  # Successfully reached goal
                successes += 1
        
        total_steps += steps
    
    average_steps = total_steps / num_episodes
    success_rate = successes / num_episodes
    
    return average_steps, success_rate

def save_results(average_steps, success_rate, training_episodes, filename='/workspace/policy_results.txt'):
    """Save evaluation results to file"""
    with open(filename, 'w') as f:
        f.write(f"average_steps: {average_steps:.2f}\n")
        f.write(f"success_rate: {success_rate:.2f}\n")
        f.write(f"training_episodes: {training_episodes}\n")

def main():
    """Main execution function"""
    print("Initializing GridWorld environment...")
    env = GridWorld()
    
    print("Initializing Q-table...")
    q_table = initialize_q_table()
    
    print(f"Starting training for {num_training_episodes} episodes...")
    q_table = train(env, q_table)
    
    print("\nTraining completed!")
    print("\nEvaluating trained policy over 100 test episodes...")
    average_steps, success_rate = evaluate(env, q_table, num_episodes=100)
    
    print(f"\nEvaluation Results:")
    print(f"Average Steps: {average_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Training Episodes: {num_training_episodes}")
    
    # Save results to file
    save_results(average_steps, success_rate, num_training_episodes)
    print(f"\nResults saved to /workspace/policy_results.txt")
    
    # Check if success criteria met
    if success_rate == 1.0 and average_steps < 15.0:
        print("\n✓ SUCCESS: Agent meets performance criteria!")
    else:
        print(f"\n✗ Performance criteria not met.")
        if success_rate < 1.0:
            print(f"  - Success rate: {success_rate:.2f} (needs to be 1.00)")
        if average_steps >= 15.0:
            print(f"  - Average steps: {average_steps:.2f} (needs to be < 15.0)")

if __name__ == "__main__":
    main()