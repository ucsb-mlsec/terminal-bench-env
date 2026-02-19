const express = require('express');
const { ApolloServer } = require('apollo-server-express');
const fs = require('fs');
const resolvers = require('./resolvers');

const typeDefs = fs.readFileSync('./schema.graphql', 'utf-8');

const server = new ApolloServer({
  typeDefs,
  resolvers
});

const app = express();

async function startServer() {
  await server.start();
  server.applyMiddleware({ app, path: '/graphql' });
  
  app.listen(4000, () => {
    console.log('GraphQL server running at http://localhost:4000/graphql');
  });
}

startServer();