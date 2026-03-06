#!/usr/bin/env python3

class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.current_pos = list(self.start_pos)
        
    def reset(self):
        self.current_pos = list(self.start_pos)
        return tuple(self.current_pos)
    
    def step(self, action):
        row, col = self.current_pos
        
        # Action 0: up (decrease row)
        if action == 0:
            if row > 0:
                row -= 1
        # Action 1: down (increase row)
        elif action == 1:
            if row < self.grid_size - 1:
                row += 1
        # Action 2: left (decrease col)
        elif action == 2:
            if col > 0:
                col -= 1
        # Action 3: right (increase col)
        elif action == 3:
            if col < self.grid_size - 1:
                col += 1
        
        self.current_pos = [row, col]
        next_state = tuple(self.current_pos)
        
        # Check if goal reached
        done = (next_state == self.goal_pos)
        
        # Calculate reward
        if done:
            reward = 10.0
        else:
            reward = -0.1
        
        info = {}
        
        return next_state, reward, done, info
    
    def get_state(self):
        return tuple(self.current_pos)
    
    def render(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == tuple(self.current_pos):
                    print('A', end=' ')
                elif (i, j) == self.goal_pos:
                    print('G', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()