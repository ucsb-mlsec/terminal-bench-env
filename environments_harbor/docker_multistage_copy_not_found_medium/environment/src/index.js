import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/main.css';

// Initialize application configuration
const config = {
  apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:3000',
  environment: process.env.NODE_ENV || 'development',
  version: '1.0.0'
};

console.log('Starting application...');
console.log('Environment:', config.environment);
console.log('API URL:', config.apiUrl);

// Error boundary for catching runtime errors
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Application error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong.</h1>;
    }
    return this.props.children;
  }
}

// Render the application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App config={config} />
    </ErrorBoundary>
  </React.StrictMode>
);

console.log('Application initialized successfully');