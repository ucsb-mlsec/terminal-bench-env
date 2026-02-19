exports.up = async function(knex) {
  return knex.schema.createTable('users', function(table) {
    table.increments('id').primary();
    table.string('username').unique();
    table.string('email');
  });
};

exports.down = async function(knex) {
  return knex.schema.dropTable('users');
};