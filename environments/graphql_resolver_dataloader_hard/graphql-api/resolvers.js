const { getProducts, getCategoryById } = require('./database.js');

module.exports = {
  Query: {
    products: async (parent, args) => {
      return await getProducts(args.limit);
    }
  },
  Product: {
    category: async (parent) => {
      return await getCategoryById(parent.category_id);
    }
  }
};