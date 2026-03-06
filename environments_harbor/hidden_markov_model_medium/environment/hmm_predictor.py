#!/usr/bin/env python3

import json
import math
from collections import defaultdict

class WeatherHMM:
    def __init__(self):
        """Initialize the HMM with weather states"""
        self.states = ['SUNNY', 'CLOUDY', 'RAINY']
        self.transition_prob = {}
        self.emission_prob = {}
        self.initial_prob = {}
        
    def train(self, data_file='weather_data.json'):
        """
        Train the HMM by calculating transition and emission probabilities
        from historical weather data
        """
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Count transitions between states
        transition_counts = defaultdict(lambda: defaultdict(int))
        state_counts = defaultdict(int)
        initial_counts = defaultdict(int)
        
        # Store observations for each state (temp, humidity)
        state_observations = defaultdict(list)
        
        for sequence in data['sequences']:
            states = sequence['states']
            observations = sequence['observations']
            
            # Count initial state
            initial_counts[states[0]] += 1
            
            # Count transitions
            for i in range(len(states) - 1):
                transition_counts[states[i]][states[i+1]] += 1
                state_counts[states[i]] += 1
            
            # Count last state
            state_counts[states[-1]] += 1
            
            # Store observations for emission probability
            for i, state in enumerate(states):
                obs = observations[i]
                state_observations[state].append((obs['temperature'], obs['humidity']))
        
        # Calculate initial probabilities
        total_initial = sum(initial_counts.values())
        for state in self.states:
            # BUG: Not checking if total_initial is zero
            self.initial_prob[state] = initial_counts[state] / total_initial
        
        # Calculate transition probabilities
        for state in self.states:
            self.transition_prob[state] = {}
            for next_state in self.states:
                # BUG: Division by zero possible if state_counts[state] is 0
                self.transition_prob[state][next_state] = transition_counts[state][next_state] / state_counts[state]
        
        # Calculate emission probabilities using Gaussian distribution parameters
        self.emission_prob = {}
        for state in self.states:
            observations = state_observations[state]
            if observations:
                temps = [obs[0] for obs in observations]
                humids = [obs[1] for obs in observations]
                
                temp_mean = sum(temps) / len(temps)
                humid_mean = sum(humids) / len(humids)
                
                temp_var = sum((t - temp_mean)**2 for t in temps) / len(temps)
                humid_var = sum((h - humid_mean)**2 for h in humids) / len(humids)
                
                self.emission_prob[state] = {
                    'temp_mean': temp_mean,
                    'temp_var': temp_var,
                    'humid_mean': humid_mean,
                    'humid_var': humid_var
                }
    
    def _emission_probability(self, state, observation):
        """
        Calculate emission probability for an observation given a state
        using Gaussian distribution
        """
        params = self.emission_prob[state]
        temp = observation['temperature']
        humid = observation['humidity']
        
        # Gaussian probability density function
        temp_prob = math.exp(-0.5 * ((temp - params['temp_mean'])**2) / params['temp_var'])
        # BUG: Missing normalization factor and incorrect variance handling
        temp_prob = temp_prob / math.sqrt(2 * math.pi * params['temp_var'])
        
        humid_prob = math.exp(-0.5 * ((humid - params['humid_mean'])**2) / params['humid_var'])
        humid_prob = humid_prob / math.sqrt(2 * math.pi * params['humid_var'])
        
        # BUG: Not handling case where probability could be 0
        return temp_prob * humid_prob
    
    def predict(self, observations):
        """
        Use Viterbi algorithm to find most likely state sequence
        """
        T = len(observations)
        N = len(self.states)
        
        # Initialize Viterbi tables
        viterbi = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]
        
        # Initialization step
        for state in self.states:
            emission = self._emission_probability(state, observations[0])
            # BUG: Taking log of potentially zero value causing -inf
            viterbi[0][state] = math.log(self.initial_prob[state]) + math.log(emission)
            backpointer[0][state] = None
        
        # Recursion step
        for t in range(1, T):
            for state in self.states:
                max_prob = float('-inf')
                max_state = None
                
                for prev_state in self.states:
                    # BUG: Not checking if transition_prob is zero before log
                    trans_prob = math.log(self.transition_prob[prev_state][state])
                    prob = viterbi[t-1][prev_state] + trans_prob
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_state
                
                emission = self._emission_probability(state, observations[t])
                # BUG: Log of zero causing issues
                viterbi[t][state] = max_prob + math.log(emission)
                backpointer[t][state] = max_state
        
        # Termination - find best final state
        best_prob = float('-inf')
        best_state = None
        for state in self.states:
            if viterbi[T-1][state] > best_prob:
                best_prob = viterbi[T-1][state]
                best_state = state
        
        # Backtrack to find best path
        best_path = [best_state]
        for t in range(T-1, 0, -1):
            best_state = backpointer[t][best_state]
            best_path.insert(0, best_state)
        
        return best_path

if __name__ == '__main__':
    # Initialize and train the HMM
    hmm = WeatherHMM()
    hmm.train('/workspace/weather_hmm/weather_data.json')
    
    # Load test sequences
    with open('/workspace/weather_hmm/test_sequences.json', 'r') as f:
        test_data = json.load(f)
    
    # Make predictions for each test case
    predictions = {}
    for test_case in test_data['test_cases']:
        test_id = test_case['id']
        observations = test_case['observations']
        
        predicted_states = hmm.predict(observations)
        predictions[test_id] = predicted_states
    
    # Write predictions to file
    with open('/workspace/predictions.txt', 'w') as f:
        for test_id in sorted(predictions.keys()):
            states_str = ','.join(predictions[test_id])
            f.write(f"{test_id}: {states_str}\n")
