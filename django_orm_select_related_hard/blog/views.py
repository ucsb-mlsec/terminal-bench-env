from rest_framework import generics
from .models import Article
from .serializers import ArticleSerializer


class ArticleListView(generics.ListAPIView):
    serializer_class = ArticleSerializer
    
    def get_queryset(self):
        return Article.objects.all()