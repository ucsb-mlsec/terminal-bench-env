from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.main_dashboard, name='main'),
    path('analytics/', views.analytics, name='analytics'),
    path('reports/', views.reports, name='reports'),
    path('detail/', views.DashboardDetailView.as_view(), name='detail'),
    path('settings/', views.settings_page, name='settings'),
]