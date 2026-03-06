#!/usr/bin/env python3

import pika
import json
from datetime import datetime

def callback(ch, method, properties, body):
    alert = json.loads(body)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_message = f"[{timestamp}] Station 2 received alert: {alert}\n"
    
    with open('station_2.log', 'a') as f:
        f.write(log_message)
    
    print(f"Station 2: Received alert - {alert}")

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    
    channel.exchange_declare(exchange='alerts', exchange_type='direct')
    
    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue
    
    channel.queue_bind(exchange='alerts', queue=queue_name)
    
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    
    print("Station 2: Waiting for alerts...")
    
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\nStation 2: Shutting down...")
        channel.stop_consuming()
    
    connection.close()

if __name__ == '__main__':
    main()