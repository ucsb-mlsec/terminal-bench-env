#!/usr/bin/env python3

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Author(Base):
    __tablename__ = 'authors'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    posts = relationship('Post', back_populates='author')

class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    author_id = Column(Integer, ForeignKey('authors.id'))
    author = relationship('Author', back_populates='posts')
    comments = relationship('Comment', back_populates='post')

class Comment(Base):
    __tablename__ = 'comments'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    post_id = Column(Integer, ForeignKey('posts.id'))
    post = relationship('Post', back_populates='comments')

# Create engine and session
engine = create_engine('sqlite:////workspace/blog.db')
Session = sessionmaker(bind=engine)
session = Session()

# Query all posts (this will trigger N+1 problem)
posts = session.query(Post).all()

# Create list with post data - this triggers lazy loading for each post
post_data = []
for post in posts:
    # Accessing author.name triggers a separate query for each post
    author_name = post.author.name
    # Accessing len(comments) triggers a separate query for each post
    comment_count = len(post.comments)
    post_data.append({
        'title': post.title,
        'author': author_name,
        'comments': comment_count
    })

# Sort by comment count and get top 10
post_data.sort(key=lambda x: x['comments'], reverse=True)
top_posts = post_data[:10]

# Print results
for post in top_posts:
    print(f"Post: {post['title']} | Author: {post['author']} | Comments: {post['comments']}")

session.close()