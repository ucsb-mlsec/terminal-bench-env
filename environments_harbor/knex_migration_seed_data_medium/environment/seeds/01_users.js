exports.seed = async function(knex) {
  await knex('users').insert([
    { username: 'john_doe', email: 'john@example.com' },
    { username: 'jane_smith', email: 'jane@example.com' },
    { username: 'bob_wilson', email: 'bob@example.com' },
    { username: 'alice_jones', email: 'alice@example.com' }
  ]);
};