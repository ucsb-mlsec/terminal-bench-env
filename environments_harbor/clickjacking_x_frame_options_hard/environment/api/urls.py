from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    path('data/', views.data_endpoint, name='data'),
    path('public/', views.public_api, name='public'),
    path('user/', views.user_info, name='user'),
    path('items/', views.DataView.as_view(), name='items'),
    path('webhooks/', views.webhooks, name='webhooks'),
    path('stats/', views.stats, name='stats'),
]