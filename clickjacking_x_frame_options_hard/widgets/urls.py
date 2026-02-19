from django.urls import path
from . import views

app_name = 'widgets'

urlpatterns = [
    path('embed/', views.embeddable_widget, name='embed'),
    path('config/', views.widget_config, name='config'),
    path('api/', views.WidgetAPIView.as_view(), name='api'),
]