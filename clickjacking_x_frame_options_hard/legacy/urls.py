from django.urls import path
from . import views

app_name = 'legacy'

urlpatterns = [
    path('old/', views.old_endpoint, name='old'),
    path('api/', views.legacy_api, name='api'),
    path('deprecated/', views.deprecated_view, name='deprecated'),
    path('data/', views.LegacyDataView.as_view(), name='data'),
]