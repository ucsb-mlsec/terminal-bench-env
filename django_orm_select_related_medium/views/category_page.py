from django.shortcuts import render, get_object_or_404
from django.db import models
from .models import Category, Post


def category_page(request, category_slug):
    """
    Display all posts in a specific category with optimized queries.
    This view demonstrates proper use of select_related and prefetch_related
    to avoid N+1 query problems.
    """
    # Get the category object
    category = get_object_or_404(Category, slug=category_slug)
    
    # Retrieve posts with optimized query loading
    # - select_related for ForeignKey (author, category)
    # - prefetch_related for ManyToMany (tags)
    posts = Post.objects.filter(
        category=category,
        published=True
    ).select_related(
        'author',
        'category'
    ).prefetch_related(
        'tags'
    ).order_by('-created_at')
    
    # Build context data
    context = {
        'category': category,
        'posts': posts,
        'post_count': posts.count(),
    }
    
    # Additional metadata for each post is now accessible without extra queries
    # because we've preloaded the relationships:
    # - post.author.username (loaded via select_related)
    # - post.category.name (loaded via select_related)
    # - post.tags.all() (loaded via prefetch_related)
    
    return render(request, 'categories/category_page.html', context)