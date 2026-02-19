#!/usr/bin/env python3

import random

def main():
    # Set seed for consistent testing
    random.seed(42)
    
    # Generate the target number
    target_number = random.randint(1, 100)
    
    # Print welcome messages
    print('Welcome to Number Guessing Game!')
    print('I'm thinking of a number between 1 and 100')
    
    # Initialize guess counter
    guess_count = 0
    
    # Main game loop
    while True:
        # Prompt for guess
        guess_input = input('Enter your guess: ')
        guess = int(guess_input)
        guess_count += 1
        
        # Check the guess
        if guess < target_number:
            print('Higher!')
        elif guess > target_number:
            print('Lower!')
        else:
            print(f'Correct! You won in {guess_count} guesses!')
            break

if __name__ == '__main__':
    main()