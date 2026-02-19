exports.seed = async function(knex) {
  await knex('posts').insert([
    {
      user_id: 1,
      title: 'First Post',
      content: 'This is my very first post on this platform. I am excited to share my thoughts and ideas with everyone here. Looking forward to engaging discussions and meeting new people.'
    },
    {
      user_id: 2,
      title: 'My Journey',
      content: 'Today I want to share my journey into web development. It has been a challenging but rewarding experience. From learning HTML and CSS to mastering JavaScript frameworks, every step has taught me something valuable.'
    },
    {
      user_id: 3,
      title: 'Tech Talk',
      content: 'Let us discuss the latest trends in technology. Artificial intelligence and machine learning are transforming industries at an unprecedented pace. What are your thoughts on these developments?'
    },
    {
      user_id: 1,
      title: 'Weekend Plans',
      content: 'Planning to work on some side projects this weekend. I have been meaning to build a portfolio website and experiment with some new libraries. Anyone else working on interesting projects?'
    },
    {
      user_id: 2,
      title: 'Book Recommendations',
      content: 'Just finished reading an amazing book on software architecture. The insights on designing scalable systems were invaluable. If anyone is interested in recommendations, feel free to ask!'
    }
  ]);
};