#!/usr/bin/env python3

import sys

def main():
    # Prompt 1
    response1 = input('Enter deployment environment: ')
    if response1 != 'production':
        print('Error: Invalid deployment environment')
        sys.exit(1)
    print('Validated: deployment environment')
    
    # Prompt 2
    response2 = input('Enter application version: ')
    if response2 != '2.5.1':
        print('Error: Invalid application version')
        sys.exit(1)
    print('Validated: application version')
    
    # Prompt 3
    response3 = input('Confirm deployment (yes/no): ')
    if response3 != 'yes':
        print('Error: Invalid confirmation')
        sys.exit(1)
    print('Validated: confirmation')
    
    print('All validations passed!')
    sys.exit(0)

if __name__ == '__main__':
    main()