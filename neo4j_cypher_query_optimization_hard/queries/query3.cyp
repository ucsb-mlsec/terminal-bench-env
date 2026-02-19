MATCH (targetUser:User {userId: 'user123'})
MATCH (targetUser)-[:FOLLOWS]->(followed:User)
MATCH (followed)-[:LIKES]->(post:Post)
WHERE NOT EXISTS((targetUser)-[:LIKES]->(post))
MATCH (post)-[:TAGGED_WITH]->(topic:Topic)
OPTIONAL MATCH (targetUser)-[:POSTED]->(userPost:Post)-[:TAGGED_WITH]->(topic)
WITH post, 
     count(DISTINCT followed) as friendLikes,
     count(DISTINCT topic) as topicMatch
RETURN post.postId, 
       post.title, 
       friendLikes, 
       topicMatch, 
       (friendLikes + topicMatch) as relevance
ORDER BY relevance DESC
LIMIT 10