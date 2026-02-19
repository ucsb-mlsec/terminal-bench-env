const request = require('supertest');
const { app, server } = require('./server');
const database = require('./database');

beforeAll(async () => {
  await database.initialize();
  await new Promise((resolve) => {
    server.listen(3001, () => {
      console.log('Test server started on port 3001');
      resolve();
    });
  });
});

describe('Payment Service API', () => {
  test('should return health status', async () => {
    const response = await request(app).get('/health');
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('status', 'healthy');
  });

  test('should create a payment', async () => {
    const paymentData = {
      amount: 100.50,
      currency: 'USD',
      description: 'Test payment'
    };
    
    const response = await request(app)
      .post('/api/payments')
      .send(paymentData);
    
    expect(response.status).toBe(201);
    expect(response.body).toHaveProperty('id');
    expect(response.body.amount).toBe(paymentData.amount);
  });

  test('should retrieve a payment', async () => {
    const response = await request(app).get('/api/payments/1');
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('id', 1);
  });
});