from django.shortcuts import render
from django.db import models


class Author(models.Model):
    """Author model with basic information."""
    name = models.CharField(max_length=100)
    email = models.EmailField()
    bio = models.TextField(blank=True)
    
    class Meta:
        app_label = 'content'


class Article(models.Model):
    """Article model with foreign key to Author."""
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='articles')
    published_date = models.DateTimeField(auto_now_add=True)
    slug = models.SlugField(unique=True)
    
    class Meta:
        app_label = 'content'
        ordering = ['-published_date']


def article_list(request):
    """
    Display a list of all articles with author information.
    This view retrieves articles and accesses author data without preloading.
    """
    # Retrieve all articles from the database
    articles = Article.objects.all()
    
    # Prepare article data for display
    # Each iteration will access article.author.name and article.author.email
    # causing additional database queries for each article
    article_data = []
    for article in articles:
        article_info = {
            'title': article.title,
            'author_name': article.author.name,
            'author_email': article.author.email,
            'published': article.published_date,
            'slug': article.slug
        }
        article_data.append(article_info)
    
    context = {
        'articles': article_data,
        'total_count': len(article_data)
    }
    
    return render(request, 'articles/article_list.html', context)