const path = require('path');

module.exports = {
  entry: './src/index.js',
  
  output: {
    path: path.join(__dirname, 'dist'),
    filename: 'bundle.js'
  },
  
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-react']
          }
        }
      }
    ]
  },
  
  resolve: {
    extensions: ['.js', '.jsx'],
    modules: ['node_modules']
  },
  
  mode: 'production',
  
  target: 'web',
  
  stats: {
    colors: true,
    modules: true,
    reasons: true
  }
};