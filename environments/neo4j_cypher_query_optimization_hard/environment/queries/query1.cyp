MATCH (p:Post)
MATCH (u:User)-[posted:POSTED]->(p)
MATCH (p)<-[likes:LIKES]-()
WITH p, u, count(likes) as likeCount
RETURN p, u, likeCount
ORDER BY likeCount DESC
LIMIT 20