from rest_framework import serializers
from .models import Article, Author, Category


class AuthorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Author
        fields = ['id', 'name', 'email', 'bio']


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name', 'description']


class ArticleSerializer(serializers.ModelSerializer):
    author = AuthorSerializer(read_only=True)
    category = CategorySerializer(read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'category', 'published_date', 'updated_date']