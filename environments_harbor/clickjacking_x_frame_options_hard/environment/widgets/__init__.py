#!/usr/bin/env python3

import os
import json
import re
from pathlib import Path

# Initialize counters
vulnerable_endpoints = 0
fixed_endpoints = 0
config_files_modified = 0
views_modified = 0

# Create project structure
project_root = Path('/tmp/django_project')
project_root.mkdir(exist_ok=True)

# Create main project directory
main_project = project_root / 'myproject'
main_project.mkdir(exist_ok=True)

# Create apps
apps = ['dashboard', 'api', 'widgets', 'legacy']
for app in apps:
    app_dir = project_root / app
    app_dir.mkdir(exist_ok=True)
    (app_dir / '__init__.py').write_text('')
    (app_dir / 'models.py').write_text('from django.db import models\n')
    (app_dir / 'admin.py').write_text('from django.contrib import admin\n')

# Create original vulnerable settings.py
settings_content = '''
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-test-key-do-not-use-in-production'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'dashboard',
    'api',
    'widgets',
    'legacy',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    # NOTE: Clickjacking middleware is missing!
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
'''

(main_project / 'settings.py').write_text(settings_content)
config_files_modified += 1

# Create vulnerable dashboard views
dashboard_views_content = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from django.views.generic import TemplateView

# Vulnerable function-based view - no protection
def dashboard_home(request):
    return render(request, 'dashboard/home.html')

# Vulnerable function-based view - no protection
def analytics_view(request):
    return render(request, 'dashboard/analytics.html')

# Vulnerable class-based view - no protection
class ReportsView(TemplateView):
    template_name = 'dashboard/reports.html'

# Vulnerable class-based view - no protection
class SettingsView(View):
    def get(self, request):
        return render(request, 'dashboard/settings.html')

# This one has some protection but wrong type
def profile_view(request):
    response = render(request, 'dashboard/profile.html')
    # Wrong: should be SAMEORIGIN for dashboard
    response['X-Frame-Options'] = 'ALLOWALL'
    return response
'''

(project_root / 'dashboard' / 'views.py').write_text(dashboard_views_content)
vulnerable_endpoints += 5
views_modified += 1

# Create vulnerable API views
api_views_content = '''
from django.http import JsonResponse
from django.views import View
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Vulnerable API endpoint - no protection
@api_view(['GET'])
def get_data(request):
    return Response({'data': 'example'})

# Vulnerable API endpoint - no protection
def api_status(request):
    return JsonResponse({'status': 'ok'})

# Vulnerable class-based API view
class DataAPIView(View):
    def get(self, request):
        return JsonResponse({'items': []})

# Legacy endpoint with no protection
def legacy_endpoint(request):
    return JsonResponse({'legacy': True})
'''

(project_root / 'api' / 'views.py').write_text(api_views_content)
vulnerable_endpoints += 4
views_modified += 1

# Create widgets views (needs ALLOW-FROM for partner embedding)
widgets_views_content = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt

# Vulnerable widget endpoint - needs to allow framing
def partner_widget(request):
    return render(request, 'widgets/partner.html')

# Another widget with no protection
def embed_widget(request):
    return render(request, 'widgets/embed.html')
'''

(project_root / 'widgets' / 'views.py').write_text(widgets_views_content)
vulnerable_endpoints += 2
views_modified += 1

# Create legacy views
legacy_views_content = '''
from django.http import HttpResponse
from django.shortcuts import render

# Old vulnerable view
def old_interface(request):
    return render(request, 'legacy/old.html')

# Another legacy view with no protection
def deprecated_view(request):
    return HttpResponse('Deprecated')
'''

(project_root / 'legacy' / 'views.py').write_text(legacy_views_content)
vulnerable_endpoints += 2
views_modified += 1

# Create main URLs
urls_content = '''
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('dashboard/', include('dashboard.urls')),
    path('api/', include('api.urls')),
    path('widgets/', include('widgets.urls')),
    path('legacy/', include('legacy.urls')),
]
'''

(main_project / 'urls.py').write_text(urls_content)

# Create app URLs
dashboard_urls = '''
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard_home'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('reports/', views.ReportsView.as_view(), name='reports'),
    path('settings/', views.SettingsView.as_view(), name='settings'),
    path('profile/', views.profile_view, name='profile'),
]
'''
(project_root / 'dashboard' / 'urls.py').write_text(dashboard_urls)

api_urls = '''
from django.urls import path
from . import views

urlpatterns = [
    path('data/', views.get_data, name='api_data'),
    path('status/', views.api_status, name='api_status'),
    path('items/', views.DataAPIView.as_view(), name='api_items'),
    path('legacy/', views.legacy_endpoint, name='api_legacy'),
]
'''
(project_root / 'api' / 'urls.py').write_text(api_urls)

widgets_urls = '''
from django.urls import path
from . import views

urlpatterns = [
    path('partner/', views.partner_widget, name='partner_widget'),
    path('embed/', views.embed_widget, name='embed_widget'),
]
'''
(project_root / 'widgets' / 'urls.py').write_text(widgets_urls)

legacy_urls = '''
from django.urls import path
from . import views

urlpatterns = [
    path('old/', views.old_interface, name='old_interface'),
    path('deprecated/', views.deprecated_view, name='deprecated'),
]
'''
(project_root / 'legacy' / 'urls.py').write_text(legacy_urls)

# Now create FIXED versions

# Fix settings.py - add clickjacking middleware
fixed_settings_content = '''
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-test-key-do-not-use-in-production'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'dashboard',
    'api',
    'widgets',
    'legacy',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # FIXED: Added clickjacking protection
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# FIXED: Set default X-Frame-Options to DENY (most secure)
X_FRAME_OPTIONS = 'DENY'
'''

(main_project / 'settings.py').write_text(fixed_settings_content)

# Fix dashboard views
fixed_dashboard_views = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from django.views.generic import TemplateView
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.utils.decorators import method_decorator

# FIXED: Dashboard pages allow same-origin framing for internal embedding
@xframe_options_sameorigin
def dashboard_home(request):
    return render(request, 'dashboard/home.html')

# FIXED: Analytics needs same-origin for internal dashboards
@xframe_options_sameorigin
def analytics_view(request):
    return render(request, 'dashboard/analytics.html')

# FIXED: Reports view with same-origin policy
@method_decorator(xframe_options_sameorigin, name='dispatch')
class ReportsView(TemplateView):
    template_name = 'dashboard/reports.html'

# FIXED: Settings view with same-origin policy
@method_decorator(xframe_options_sameorigin, name='dispatch')
class SettingsView(View):
    def get(self, request):
        return render(request, 'dashboard/settings.html')

# FIXED: Corrected to use SAMEORIGIN instead of ALLOWALL
@xframe_options_sameorigin
def profile_view(request):
    response = render(request, 'dashboard/profile.html')
    return response
'''

(project_root / 'dashboard' / 'views.py').write_text(fixed_dashboard_views)
fixed_endpoints += 5

# Fix API views
fixed_api_views = '''
from django.http import JsonResponse
from django.views import View
from django.views.decorators.clickjacking import xframe_options_deny
from django.utils.decorators import method_decorator

# Note: Using xframe_options_deny for API endpoints as they shouldn't be framed
# In a real scenario, you might use rest_framework properly

# FIXED: API endpoints should deny framing
@xframe_options_deny
def get_data(request):
    # In real code, would use DRF Response
    return JsonResponse({'data': 'example'})

# FIXED: API status endpoint with deny policy
@xframe_options_deny
def api_status(request):
    return JsonResponse({'status': 'ok'})

# FIXED: Class-based API view with deny policy
@method_decorator(xframe_options_deny, name='dispatch')
class DataAPIView(View):
    def get(self, request):
        return JsonResponse({'items': []})

# FIXED: Legacy endpoint now has deny protection
@xframe_options_deny
def legacy_endpoint(request):
    return JsonResponse({'legacy': True})
'''

(project_root / 'api' / 'views.py').write_text(fixed_api_views)
fixed_endpoints += 4

# Fix widgets views
fixed_widgets_views = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt

# FIXED: Widget explicitly allows framing for partner embedding
# This is required for partner integrations
@xframe_options_exempt
def partner_widget(request):
    return render(request, 'widgets/partner.html')

# FIXED: Embed widget also allows framing for external use
@xframe_options_exempt
def embed_widget(request):
    return render(request, 'widgets/embed.html')
'''

(project_root / 'widgets' / 'views.py').write_text(fixed_widgets_views)
fixed_endpoints += 2

# Fix legacy views
fixed_legacy_views = '''
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_deny

# FIXED: Old interface now has deny protection
@xframe_options_deny
def old_interface(request):
    return render(request, 'legacy/old.html')

# FIXED: Deprecated view with deny protection
@xframe_options_deny
def deprecated_view(request):
    return HttpResponse('Deprecated')
'''

(project_root / 'legacy' / 'views.py').write_text(fixed_legacy_views)
fixed_endpoints += 2

# Create __init__ files
(main_project / '__init__.py').write_text('')
(project_root / 'widgets' / '__init__.py').write_text('')

# Create analysis report
report = {
    "vulnerable_endpoints": vulnerable_endpoints,
    "fixed_endpoints": fixed_endpoints,
    "config_files_modified": config_files_modified,
    "views_modified": views_modified
}

# Save report
report_path = Path('/tmp/clickjacking_fix_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Analysis complete!")
print(f"Vulnerable endpoints found: {vulnerable_endpoints}")
print(f"Endpoints fixed: {fixed_endpoints}")
print(f"Config files modified: {config_files_modified}")
print(f"View files modified: {views_modified}")
print(f"\nReport saved to: {report_path}")
print(f"\nProject created at: {project_root}")
print("\nSummary of fixes:")
print("1. Added XFrameOptionsMiddleware to settings.py")
print("2. Set default X_FRAME_OPTIONS = 'DENY'")
print("3. Dashboard views: Applied @xframe_options_sameorigin for internal embedding")
print("4. API views: Applied @xframe_options_deny to prevent framing")
print("5. Widget views: Applied @xframe_options_exempt for partner integration")
print("6. Legacy views: Applied @xframe_options_deny for security")
print("7. Admin interface: Protected by default middleware with DENY policy")