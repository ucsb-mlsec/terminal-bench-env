exports.up = async function(knex) {
  await knex.schema.createTable('comments', function(table) {
    table.increments('id').primary();
    table.integer('post_id').unsigned().notNullable();
    table.foreign('post_id').references('id').inTable('posts').onDelete('CASCADE');
    table.integer('user_id').unsigned().notNullable();
    table.foreign('user_id').references('id').inTable('users').onDelete('CASCADE');
    table.text('comment_text');
  });
};

exports.down = async function(knex) {
  await knex.schema.dropTable('comments');
};