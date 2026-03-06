exports.seed = async function(knex) {
  await knex('comments').insert([
    { post_id: 1, user_id: 2, comment_text: 'Great post!' },
    { post_id: 1, user_id: 3, comment_text: 'Thanks for sharing' },
    { post_id: 2, user_id: 1, comment_text: 'Interesting perspective' },
    { post_id: 2, user_id: 3, comment_text: 'I completely agree with this' },
    { post_id: 3, user_id: 1, comment_text: 'Very informative, thank you' },
    { post_id: 3, user_id: 2, comment_text: 'This is exactly what I needed' }
  ]);
};