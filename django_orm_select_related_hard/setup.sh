#!/bin/bash

# Create workspace structure
mkdir -p /workspace/blog/migrations

# Create __init__.py files
touch /workspace/blog/__init__.py
touch /workspace/blog/migrations/__init__.py

# Create settings.py
cat > /workspace/blog/settings.py << 'SETTINGS_EOF'
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'test-secret-key-for-development'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'rest_framework',
    'blog',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'blog.urls'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': '/workspace/db.sqlite3',
    }
}

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100
}

USE_TZ = True
SETTINGS_EOF

# Create models.py
cat > /workspace/blog/models.py << 'MODELS_EOF'
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    
    def __str__(self):
        return self.name

class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    
    def __str__(self):
        return self.name

class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='articles')
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='articles')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
MODELS_EOF

# Create serializers.py
cat > /workspace/blog/serializers.py << 'SERIALIZERS_EOF'
from rest_framework import serializers
from .models import Article, Author, Category

class AuthorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Author
        fields = ['id', 'name', 'email']

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name', 'description']

class ArticleSerializer(serializers.ModelSerializer):
    author = AuthorSerializer(read_only=True)
    category = CategorySerializer(read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'category', 'created_at']
SERIALIZERS_EOF

# Create views.py with inefficient implementation
cat > /workspace/blog/views.py << 'VIEWS_EOF'
from rest_framework import generics
from .models import Article
from .serializers import ArticleSerializer

class ArticleListView(generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
VIEWS_EOF

# Create urls.py
cat > /workspace/blog/urls.py << 'URLS_EOF'
from django.urls import path
from .views import ArticleListView

urlpatterns = [
    path('api/articles/', ArticleListView.as_view(), name='article-list'),
]
URLS_EOF

# Create manage.py
cat > /workspace/manage.py << 'MANAGE_EOF'
#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
MANAGE_EOF

chmod +x /workspace/manage.py

echo "=== Creating migrations ==="
cd /workspace
python manage.py makemigrations blog

echo "=== Running migrations ==="
python manage.py migrate

echo "=== Populating database with test data ==="
cat > /workspace/populate_data.py << 'POPULATE_EOF'
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog.settings')
django.setup()

from blog.models import Author, Category, Article
import random

# Create categories
categories = []
for i in range(10):
    categories.append(Category(
        name=f'Category {i+1}',
        description=f'Description for category {i+1}'
    ))
Category.objects.bulk_create(categories)
print(f"Created {len(categories)} categories")

# Create authors
authors = []
for i in range(20):
    authors.append(Author(
        name=f'Author {i+1}',
        email=f'author{i+1}@example.com'
    ))
Author.objects.bulk_create(authors)
print(f"Created {len(authors)} authors")

# Fetch created objects for foreign keys
all_categories = list(Category.objects.all())
all_authors = list(Author.objects.all())

# Create articles
articles = []
for i in range(100):
    articles.append(Article(
        title=f'Article {i+1}',
        content=f'Content for article {i+1}. This is a longer text to simulate real article content.',
        author=random.choice(all_authors),
        category=random.choice(all_categories)
    ))
Article.objects.bulk_create(articles)
print(f"Created {len(articles)} articles")

print("Database population complete!")
POPULATE_EOF

python /workspace/populate_data.py

echo "=== Measuring baseline queries ==="
cat > /workspace/measure_queries.py << 'MEASURE_EOF'
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog.settings')
django.setup()

from django.test.utils import override_settings
from django.db import connection, reset_queries
from django.test import RequestFactory
from blog.views import ArticleListView

# Enable query logging
with override_settings(DEBUG=True):
    reset_queries()
    
    factory = RequestFactory()
    request = factory.get('/api/articles/')
    
    view = ArticleListView.as_view()
    response = view(request)
    
    # Force evaluation of response
    _ = response.render()
    
    query_count = len(connection.queries)
    print(query_count)
MEASURE_EOF

BASELINE_QUERIES=$(python /workspace/measure_queries.py)
echo $BASELINE_QUERIES > /workspace/baseline_queries.txt

echo "=== Setup Complete ==="
echo "Baseline queries: $BASELINE_QUERIES"
echo "Baseline saved to /workspace/baseline_queries.txt"
echo ""
echo "Next steps:"
echo "1. Optimize /workspace/blog/views.py"
echo "2. Run: cd /workspace/blog && python ../measure_queries.py"
echo "3. Create /workspace/solution.json with before/after query counts"