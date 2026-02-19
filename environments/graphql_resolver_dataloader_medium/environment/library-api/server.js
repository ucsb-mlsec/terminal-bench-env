const express = require('express');
const { graphqlHTTP } = require('express-graphql');
const { GraphQLSchema, GraphQLObjectType, GraphQLString, GraphQLInt, GraphQLList, GraphQLNonNull } = require('graphql');

// Mock data
const authors = [
  { id: 1, name: 'Jane Austen', bio: 'English novelist known for her six major novels' },
  { id: 2, name: 'George Orwell', bio: 'English novelist and essayist, journalist and critic' },
  { id: 3, name: 'Agatha Christie', bio: 'English writer known for her detective novels' }
];

const books = [
  { id: 1, title: 'Pride and Prejudice', authorId: 1, publishedYear: 1813 },
  { id: 2, title: 'Sense and Sensibility', authorId: 1, publishedYear: 1811 },
  { id: 3, title: 'Emma', authorId: 1, publishedYear: 1815 },
  { id: 4, title: 'Mansfield Park', authorId: 1, publishedYear: 1814 },
  { id: 5, title: '1984', authorId: 2, publishedYear: 1949 },
  { id: 6, title: 'Animal Farm', authorId: 2, publishedYear: 1945 },
  { id: 7, title: 'Murder on the Orient Express', authorId: 3, publishedYear: 1934 },
  { id: 8, title: 'Death on the Nile', authorId: 3, publishedYear: 1937 },
  { id: 9, title: 'The Murder of Roger Ackroyd', authorId: 3, publishedYear: 1926 },
  { id: 10, title: 'And Then There Were None', authorId: 3, publishedYear: 1939 }
];

// Tracking variable
let authorFetchCount = 0;

// Define Author type
const AuthorType = new GraphQLObjectType({
  name: 'Author',
  fields: {
    id: { type: new GraphQLNonNull(GraphQLInt) },
    name: { type: new GraphQLNonNull(GraphQLString) },
    bio: { type: new GraphQLNonNull(GraphQLString) }
  }
});

// Define Book type
const BookType = new GraphQLObjectType({
  name: 'Book',
  fields: {
    id: { type: new GraphQLNonNull(GraphQLInt) },
    title: { type: new GraphQLNonNull(GraphQLString) },
    publishedYear: { type: new GraphQLNonNull(GraphQLInt) },
    author: {
      type: AuthorType,
      resolve: (book) => {
        // Increment fetch count - demonstrating N+1 problem
        authorFetchCount++;
        // Fetch author individually for each book
        return authors.find(author => author.id === book.authorId);
      }
    }
  }
});

// Define Query type
const QueryType = new GraphQLObjectType({
  name: 'Query',
  fields: {
    books: {
      type: new GraphQLList(BookType),
      resolve: () => books
    }
  }
});

// Create schema
const schema = new GraphQLSchema({
  query: QueryType
});

// Create Express app
const app = express();

app.use(express.json());

// GraphQL endpoint
app.use('/graphql', graphqlHTTP({
  schema: schema,
  graphiql: true
}));

// REST endpoint to get fetch count
app.get('/fetch-count', (req, res) => {
  res.json({ authorFetchCount });
});

// REST endpoint to reset fetch count
app.post('/reset-count', (req, res) => {
  authorFetchCount = 0;
  res.json({ authorFetchCount });
});

// Start server
const PORT = 4000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}/graphql`);
});