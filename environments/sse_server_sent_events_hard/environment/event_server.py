from flask import Flask, Response, request
import time
import json

app = Flask(__name__)

event_counter = 0
active_clients = []

def generate_events(last_id):
    global event_counter
    client_buffer = []
    active_clients.append(client_buffer)
    
    try:
        # Convert last_id to integer, default to 0 if invalid
        last_id_num = int(last_id)
        
        while True:
            event_counter += 1
            
            # Skip events that were already sent to this client
            if event_counter <= last_id_num:
                continue
            
            event_data = {
                'message': f'Alert {event_counter}',
                'severity': 'medium'
            }
            
            # Include the id: field in the SSE format
            yield f"id: {event_counter}\n"
            yield f"data: {json.dumps(event_data)}\n\n"
            time.sleep(2)
    finally:
        # Remove client buffer when client disconnects
        if client_buffer in active_clients:
            active_clients.remove(client_buffer)

@app.route('/events')
def events():
    last_id = request.headers.get('Last-Event-ID', '0')
    return Response(generate_events(last_id), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)