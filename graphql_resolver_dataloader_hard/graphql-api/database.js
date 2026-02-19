const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('./products.db');

let queryCount = 0;

function resetQueryCount() {
  queryCount = 0;
}

function getQueryCount() {
  return queryCount;
}

function getProducts(limit) {
  return new Promise((resolve, reject) => {
    queryCount++;
    let query = 'SELECT * FROM products';
    const params = [];
    
    if (limit) {
      query += ' LIMIT ?';
      params.push(limit);
    }
    
    db.all(query, params, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows);
      }
    });
  });
}

function getCategoryById(id) {
  return new Promise((resolve, reject) => {
    queryCount++;
    db.get('SELECT * FROM categories WHERE id = ?', [id], (err, row) => {
      if (err) {
        reject(err);
      } else {
        resolve(row);
      }
    });
  });
}

function getCategoriesByIds(ids) {
  return new Promise((resolve, reject) => {
    queryCount++;
    const placeholders = ids.map(() => '?').join(',');
    const query = `SELECT * FROM categories WHERE id IN (${placeholders})`;
    
    db.all(query, ids, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows);
      }
    });
  });
}

module.exports = {
  resetQueryCount,
  getQueryCount,
  getProducts,
  getCategoryById,
  getCategoriesByIds
};