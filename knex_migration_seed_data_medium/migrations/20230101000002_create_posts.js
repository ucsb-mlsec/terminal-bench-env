exports.up = async function(knex) {
  await knex.schema.createTable('posts', function(table) {
    table.increments('id').primary();
    table.integer('user_id').notNullable();
    table.foreign('user_id').references('users.id').onDelete('CASCADE');
    table.string('title').notNullable();
    table.text('content');
  });
};

exports.down = async function(knex) {
  await knex.schema.dropTable('posts');
};