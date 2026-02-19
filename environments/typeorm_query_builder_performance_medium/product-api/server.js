require('reflect-metadata');
const express = require('express');
const { DataSource } = require('typeorm');
const Product = require('./entities/Product');
const Category = require('./entities/Category');

const app = express();
app.use(express.json());

const AppDataSource = new DataSource({
  type: 'sqlite',
  database: './products.db',
  entities: [Product, Category],
  synchronize: false,
  logging: true
});

app.get('/api/products', async (req, res) => {
  try {
    const productRepository = AppDataSource.getRepository(Product);
    
    // BUG: Fetching products without eager loading relations
    const products = await productRepository.find();
    
    // BUG: Manually accessing category for each product causes N+1 queries
    const productsWithCategories = [];
    for (const product of products) {
      // This triggers a separate query for each product's category
      const category = product.category;
      productsWithCategories.push({
        id: product.id,
        name: product.name,
        price: product.price,
        description: product.description,
        category: {
          id: category.id,
          name: category.name,
          description: category.description
        }
      });
    }
    
    res.json(productsWithCategories);
  } catch (error) {
    console.error('Error fetching products:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

let server;

const startServer = async () => {
  await AppDataSource.initialize();
  console.log('Data Source has been initialized!');
  
  server = app.listen(3000, () => {
    console.log('Server running on port 3000');
  });
};

const stopServer = async () => {
  if (server) {
    server.close();
  }
  if (AppDataSource.isInitialized) {
    await AppDataSource.destroy();
  }
};

if (require.main === module) {
  startServer().catch(error => {
    console.error('Error starting server:', error);
    process.exit(1);
  });
}

module.exports = { app, AppDataSource, startServer, stopServer };