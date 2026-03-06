const config = {
  development: {
    host: 'localhost',
    port: 5432,
    database: 'microservices_dev',
    user: 'dev_user',
    password: 'dev_password',
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
  },
  production: {
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_PORT || 5432,
    database: process.env.DB_NAME || 'microservices_prod',
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    max: 50,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
    ssl: {
      rejectUnauthorized: false
    }
  }
};

function getConfig() {
  const env = process.env.NODE_ENV || 'development';
  return config[env] || config.development;
}

module.exports = {
  config,
  getConfig
};