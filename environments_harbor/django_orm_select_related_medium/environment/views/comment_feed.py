from django.shortcuts import render
from django.db import models
from .models import Comment

def comment_feed(request):
    """
    Display a feed of recent comments with user and post information.
    
    This view retrieves the 50 most recent comments and displays them
    along with related user and post data.
    """
    # Retrieve recent comments from the database
    comments = Comment.objects.order_by('-created_at')[:50]
    
    # Prepare comment data for display
    comment_list = []
    for comment in comments:
        # Access related user data through foreign key
        username = comment.user.username
        user_email = comment.user.email
        user_profile_pic = comment.user.profile.picture_url
        
        # Access related post data through foreign key
        post_title = comment.post.title
        post_slug = comment.post.slug
        post_author = comment.post.author.username
        
        # Build comment display data
        comment_data = {
            'id': comment.id,
            'text': comment.text,
            'created_at': comment.created_at,
            'username': username,
            'user_email': user_email,
            'user_profile_pic': user_profile_pic,
            'post_title': post_title,
            'post_slug': post_slug,
            'post_author': post_author,
            'likes_count': comment.likes_count,
        }
        comment_list.append(comment_data)
    
    context = {
        'comments': comment_list,
        'total_count': len(comment_list),
    }
    
    return render(request, 'comments/feed.html', context)