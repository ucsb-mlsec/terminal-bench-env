#!/usr/bin/env python3

import json
import os
import sys

def process_message(message):
    message_id = message['message_id']
    amount = message['amount']
    account_id = message['account_id']
    
    with open('/workspace/output/transactions.txt', 'a') as f:
        f.write(f'Transaction: ID={message_id}, Account={account_id}, Amount=${amount}\n')

def main():
    if not os.path.exists('/workspace/output/'):
        os.makedirs('/workspace/output/')
    
    with open('/workspace/test_queue/messages.json', 'r') as f:
        messages = json.load(f)
    
    for message in messages:
        message_id = message['message_id']
        print(f'Processing message {message_id}')
        process_message(message)
    
    print(f'Total messages processed: {len(messages)}')

if __name__ == '__main__':
    main()