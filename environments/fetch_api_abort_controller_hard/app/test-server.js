const http = require('http');

const server = http.createServer((req, res) => {
  console.log(`${req.method} ${req.url}`);
  
  res.setHeader('Content-Type', 'application/json');
  
  if (req.url === '/fast') {
    res.statusCode = 200;
    res.end(JSON.stringify({
      status: 'ok',
      endpoint: 'fast',
      delay: 0
    }));
  } else if (req.url === '/medium') {
    setTimeout(() => {
      res.statusCode = 200;
      res.end(JSON.stringify({
        status: 'ok',
        endpoint: 'medium',
        delay: 1000
      }));
    }, 1000);
  } else if (req.url === '/slow') {
    setTimeout(() => {
      res.statusCode = 200;
      res.end(JSON.stringify({
        status: 'ok',
        endpoint: 'slow',
        delay: 3000
      }));
    }, 3000);
  } else {
    res.statusCode = 404;
    res.end(JSON.stringify({
      error: 'Not found'
    }));
  }
});

server.listen(3000, () => {
  console.log('Test server running on port 3000');
});