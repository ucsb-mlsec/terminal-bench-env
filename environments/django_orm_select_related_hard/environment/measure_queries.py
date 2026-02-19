#!/usr/bin/env python
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog.settings')
sys.path.insert(0, '/workspace')
django.setup()

from django.test import RequestFactory
from django.db import connection, reset_queries
from django.conf import settings
from blog.views import ArticleListView
from blog.serializers import ArticleSerializer

# Enable query logging
settings.DEBUG = True

# Reset queries
reset_queries()

# Create a mock GET request
factory = RequestFactory()
request = factory.get('/api/articles/')

# Instantiate view and get queryset
view = ArticleListView()
view.request = request
queryset = view.get_queryset()

# Force evaluation and serialization (simulate full view execution)
articles_list = list(queryset)
serializer = ArticleSerializer(articles_list, many=True)
data = serializer.data

# Count and print total queries
query_count = len(connection.queries)
print(query_count)