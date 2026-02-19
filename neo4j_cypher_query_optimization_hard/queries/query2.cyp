MATCH (u:User)
MATCH (u)-[:POSTED]->(p:Post)
MATCH (p)-[:TAGGED_WITH]->(t:Topic)
MATCH (u)<-[:FOLLOWS]-(follower:User)
MATCH (p)<-[:LIKES]-(liker:User)
WHERE t.name = 'Technology'
WITH u, count(DISTINCT follower) as followerCount, count(DISTINCT p) as postCount, count(liker) as totalLikes
RETURN u.userId, u.name, u.email, followerCount, postCount, totalLikes, (followerCount * 10 + totalLikes * 5 + postCount * 3) as influenceScore
ORDER BY influenceScore DESC
LIMIT 15