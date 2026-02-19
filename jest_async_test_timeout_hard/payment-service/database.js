const sqlite3 = require('sqlite3').verbose();

let db = null;

const database = {
  initialize: function() {
    return new Promise((resolve, reject) => {
      db = new sqlite3.Database(':memory:', (err) => {
        if (err) {
          return reject(err);
        }
        
        db.run(`
          CREATE TABLE payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            status TEXT NOT NULL
          )
        `, (err) => {
          if (err) {
            return reject(err);
          }
          resolve();
        });
      });
    });
  },

  close: function() {
    return new Promise((resolve, reject) => {
      if (db) {
        db.close((err) => {
          if (err) {
            return reject(err);
          }
          db = null;
          resolve();
        });
      } else {
        resolve();
      }
    });
  },

  createPayment: function(amount, status) {
    return new Promise((resolve, reject) => {
      if (!db) {
        return reject(new Error('Database not initialized'));
      }
      
      db.run(
        'INSERT INTO payments (amount, status) VALUES (?, ?)',
        [amount, status],
        function(err) {
          if (err) {
            return reject(err);
          }
          resolve({ id: this.lastID, amount, status });
        }
      );
    });
  },

  getPayment: function(id) {
    return new Promise((resolve, reject) => {
      if (!db) {
        return reject(new Error('Database not initialized'));
      }
      
      db.get(
        'SELECT * FROM payments WHERE id = ?',
        [id],
        (err, row) => {
          if (err) {
            return reject(err);
          }
          resolve(row);
        }
      );
    });
  }
};

module.exports = database;