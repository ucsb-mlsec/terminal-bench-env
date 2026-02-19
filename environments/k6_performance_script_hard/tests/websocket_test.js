import ws from 'k6/ws';
import http from 'k6/http';
import { check } from 'k6';

export let options = {
    vus: 10,
    duration: '30s',
};

export default function () {
    const url = 'ws://localhost:3000/notifications';
    
    const response = ws.connect(url, function (socket) {
        socket.on('open', function open() {
            console.log('WebSocket connection established');
            
            socket.send(JSON.stringify({
                type: 'subscribe',
                channel: 'orders'
            }));
        });

        socket.on('message', function (message) {
            console.log('Received message: ' + message);
            const data = JSON.parse(message);
            
            check(data, {
                'message has type': (d) => d.type !== undefined,
                'message has payload': (d) => d.payload !== undefined,
            });
        });

        socket.on('error', function (e) {
            console.log('WebSocket error: ' + e.error());
        });

        socket.setTimeout(function () {
            console.log('Connection timeout');
        }, 30000);
    });

    check(response, {
        'WebSocket connection status is 101': (r) => r && r.status === 101,
    });
}