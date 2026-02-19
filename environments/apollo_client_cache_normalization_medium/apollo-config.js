import { ApolloClient, InMemoryCache, HttpLink } from '@apollo/client';

/**
 * Apollo Client Configuration
 * 
 * Basic setup for connecting to GraphQL API
 * TODO: Add proper cache normalization configuration
 */

// Basic cache setup without type policies
// This might cause issues with entity identification
const cache = new InMemoryCache({
  // Just using default settings for now
  addTypename: true,
});

// HTTP connection to the API
const link = new HttpLink({
  uri: 'http://localhost:4000/graphql',
  credentials: 'same-origin',
});

// Create the Apollo Client instance
const client = new ApolloClient({
  cache,
  link,
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network',
    },
  },
});

export default client;