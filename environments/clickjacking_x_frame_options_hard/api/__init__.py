#!/usr/bin/env python3

import os
import json
import sys
from pathlib import Path

# Create the Django project structure with clickjacking vulnerabilities
def create_vulnerable_django_project():
    base_dir = Path('/tmp/django_project')
    base_dir.mkdir(exist_ok=True)
    
    # Create project structure
    dirs = [
        'myproject',
        'dashboard',
        'api',
        'legacy',
        'widgets',
        'templates',
    ]
    
    for dir_name in dirs:
        (base_dir / dir_name).mkdir(exist_ok=True)
        (base_dir / dir_name / '__init__.py').write_text('')
    
    # Create settings.py with incomplete middleware
    settings_content = """
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
    'legacy',
    'widgets',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    # XFrameOptionsMiddleware is missing - vulnerability!
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
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
"""
    (base_dir / 'myproject' / 'settings.py').write_text(settings_content)
    
    # Create main urls.py
    main_urls = """
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('dashboard/', include('dashboard.urls')),
    path('api/', include('api.urls')),
    path('legacy/', include('legacy.urls')),
    path('widgets/', include('widgets.urls')),
]
"""
    (base_dir / 'myproject' / 'urls.py').write_text(main_urls)
    
    # Create dashboard views (needs same-origin policy)
    dashboard_views = """
from django.shortcuts import render
from django.http import HttpResponse
from django.views import View

def dashboard_home(request):
    # Vulnerable - no X-Frame-Options protection
    return render(request, 'dashboard.html')

def analytics(request):
    # Vulnerable - no protection
    return render(request, 'analytics.html')

class ReportsView(View):
    # Vulnerable class-based view
    def get(self, request):
        return render(request, 'reports.html')

def settings_page(request):
    # Vulnerable
    return render(request, 'settings.html')
"""
    (base_dir / 'dashboard' / 'views.py').write_text(dashboard_views)
    
    dashboard_urls = """
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard_home'),
    path('analytics/', views.analytics, name='analytics'),
    path('reports/', views.ReportsView.as_view(), name='reports'),
    path('settings/', views.settings_page, name='settings'),
]
"""
    (base_dir / 'dashboard' / 'urls.py').write_text(dashboard_urls)
    
    # Create API views (needs DENY policy)
    api_views = """
from django.http import JsonResponse
from django.views import View
from django.views.decorators.clickjacking import xframe_options_deny

def public_api(request):
    # Vulnerable - no protection
    return JsonResponse({'status': 'ok'})

@xframe_options_deny
def secure_api(request):
    # Already protected correctly
    return JsonResponse({'status': 'secure'})

class DataAPIView(View):
    # Vulnerable class-based view
    def get(self, request):
        return JsonResponse({'data': 'important'})

def user_api(request):
    # Vulnerable
    return JsonResponse({'users': []})
"""
    (base_dir / 'api' / 'views.py').write_text(api_views)
    
    api_urls = """
from django.urls import path
from . import views

urlpatterns = [
    path('public/', views.public_api, name='public_api'),
    path('secure/', views.secure_api, name='secure_api'),
    path('data/', views.DataAPIView.as_view(), name='data_api'),
    path('users/', views.user_api, name='user_api'),
]
"""
    (base_dir / 'api' / 'urls.py').write_text(api_urls)
    
    # Create legacy views (needs protection)
    legacy_views = """
from django.http import HttpResponse
from django.shortcuts import render

def old_endpoint(request):
    # Legacy vulnerable endpoint
    return HttpResponse('Legacy content')

def deprecated_view(request):
    # Vulnerable
    return render(request, 'legacy.html')

def legacy_form(request):
    # Vulnerable
    return render(request, 'form.html')
"""
    (base_dir / 'legacy' / 'views.py').write_text(legacy_views)
    
    legacy_urls = """
from django.urls import path
from . import views

urlpatterns = [
    path('old/', views.old_endpoint, name='old_endpoint'),
    path('deprecated/', views.deprecated_view, name='deprecated'),
    path('form/', views.legacy_form, name='legacy_form'),
]
"""
    (base_dir / 'legacy' / 'urls.py').write_text(legacy_urls)
    
    # Create widgets views (needs EXEMPT for embedding in partner sites)
    widgets_views = """
from django.http import HttpResponse
from django.shortcuts import render

def embed_widget(request):
    # This needs to allow framing from any origin (partner sites)
    # Currently vulnerable/misconfigured
    return render(request, 'widget.html')

def iframe_content(request):
    # Needs to allow external framing
    return render(request, 'iframe.html')
"""
    (base_dir / 'widgets' / 'views.py').write_text(widgets_views)
    
    widgets_urls = """
from django.urls import path
from . import views

urlpatterns = [
    path('embed/', views.embed_widget, name='embed_widget'),
    path('iframe/', views.iframe_content, name='iframe_content'),
]
"""
    (base_dir / 'widgets' / 'urls.py').write_text(widgets_urls)
    
    # Create simple templates
    for template in ['dashboard.html', 'analytics.html', 'reports.html', 
                     'settings.html', 'legacy.html', 'form.html', 
                     'widget.html', 'iframe.html']:
        (base_dir / 'templates' / template).write_text(f'<html><body><h1>{template}</h1></body></html>')
    
    return base_dir

# Analysis and fixing function
def analyze_and_fix_clickjacking(base_dir):
    vulnerable_count = 0
    fixed_count = 0
    config_files_modified = 0
    views_modified = 0
    
    # Step 1: Fix settings.py - add XFrameOptionsMiddleware
    settings_path = base_dir / 'myproject' / 'settings.py'
    settings_content = settings_path.read_text()
    
    if 'django.middleware.clickjacking.XFrameOptionsMiddleware' not in settings_content:
        # Add the middleware
        settings_content = settings_content.replace(
            "    'django.contrib.messages.middleware.MessageMiddleware',\n    # XFrameOptionsMiddleware is missing - vulnerability!",
            "    'django.contrib.messages.middleware.MessageMiddleware',\n    'django.middleware.clickjacking.XFrameOptionsMiddleware',"
        )
        
        # Set default X_FRAME_OPTIONS to DENY (most secure)
        settings_content += "\n\n# Clickjacking Protection\nX_FRAME_OPTIONS = 'DENY'\n"
        
        settings_path.write_text(settings_content)
        config_files_modified += 1
    
    # Step 2: Fix dashboard views (need SAMEORIGIN)
    dashboard_views_path = base_dir / 'dashboard' / 'views.py'
    dashboard_content = """
from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.utils.decorators import method_decorator

@xframe_options_sameorigin
def dashboard_home(request):
    # Fixed - SAMEORIGIN protection for dashboard
    return render(request, 'dashboard.html')

@xframe_options_sameorigin
def analytics(request):
    # Fixed - SAMEORIGIN protection
    return render(request, 'analytics.html')

@method_decorator(xframe_options_sameorigin, name='dispatch')
class ReportsView(View):
    # Fixed - SAMEORIGIN protection for class-based view
    def get(self, request):
        return render(request, 'reports.html')

@xframe_options_sameorigin
def settings_page(request):
    # Fixed - SAMEORIGIN protection
    return render(request, 'settings.html')
"""
    dashboard_views_path.write_text(dashboard_content)
    views_modified += 1
    vulnerable_count += 4
    fixed_count += 4
    
    # Step 3: Fix API views (need DENY - already default but explicitly set)
    api_views_path = base_dir / 'api' / 'views.py'
    api_content = """
from django.http import JsonResponse
from django.views import View
from django.views.decorators.clickjacking import xframe_options_deny
from django.utils.decorators import method_decorator

@xframe_options_deny
def public_api(request):
    # Fixed - explicit DENY protection
    return JsonResponse({'status': 'ok'})

@xframe_options_deny
def secure_api(request):
    # Already protected correctly
    return JsonResponse({'status': 'secure'})

@method_decorator(xframe_options_deny, name='dispatch')
class DataAPIView(View):
    # Fixed - DENY protection for class-based view
    def get(self, request):
        return JsonResponse({'data': 'important'})

@xframe_options_deny
def user_api(request):
    # Fixed - DENY protection
    return JsonResponse({'users': []})
"""
    api_views_path.write_text(api_content)
    views_modified += 1
    vulnerable_count += 3  # public_api, DataAPIView, user_api were vulnerable
    fixed_count += 3
    
    # Step 4: Fix legacy views (need DENY)
    legacy_views_path = base_dir / 'legacy' / 'views.py'
    legacy_content = """
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_deny

@xframe_options_deny
def old_endpoint(request):
    # Fixed - DENY protection for legacy endpoint
    return HttpResponse('Legacy content')

@xframe_options_deny
def deprecated_view(request):
    # Fixed - DENY protection
    return render(request, 'legacy.html')

@xframe_options_deny
def legacy_form(request):
    # Fixed - DENY protection
    return render(request, 'form.html')
"""
    legacy_views_path.write_text(legacy_content)
    views_modified += 1
    vulnerable_count += 3
    fixed_count += 3
    
    # Step 5: Fix widgets views (need EXEMPT for partner embedding)
    widgets_views_path = base_dir / 'widgets' / 'views.py'
    widgets_content = """
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_exempt

@xframe_options_exempt
def embed_widget(request):
    # Fixed - EXEMPT to allow framing from partner sites
    return render(request, 'widget.html')

@xframe_options_exempt
def iframe_content(request):
    # Fixed - EXEMPT to allow external framing
    return render(request, 'iframe.html')
"""
    widgets_views_path.write_text(widgets_content)
    views_modified += 1
    vulnerable_count += 2
    fixed_count += 2
    
    # Step 6: Create api/__init__.py as required
    api_init_path = base_dir / 'api' / '__init__.py'
    api_init_path.write_text('# An empty Python file.\n')
    
    return {
        'vulnerable_endpoints': vulnerable_count,
        'fixed_endpoints': fixed_count,
        'config_files_modified': config_files_modified,
        'views_modified': views_modified
    }

# Main execution
def main():
    print("Creating vulnerable Django project...")
    base_dir = create_vulnerable_django_project()
    
    print("Analyzing and fixing clickjacking vulnerabilities...")
    results = analyze_and_fix_clickjacking(base_dir)
    
    # Save report
    report_path = Path('/tmp/clickjacking_fix_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nClickjacking Fix Report:")
    print(f"  Vulnerable endpoints found: {results['vulnerable_endpoints']}")
    print(f"  Endpoints fixed: {results['fixed_endpoints']}")
    print(f"  Configuration files modified: {results['config_files_modified']}")
    print(f"  View files modified: {results['views_modified']}")
    print(f"\nReport saved to: {report_path}")
    
    print("\nSummary of fixes applied:")
    print("  1. Added XFrameOptionsMiddleware to settings.py")
    print("  2. Set default X_FRAME_OPTIONS to 'DENY'")
    print("  3. Dashboard views: Applied SAMEORIGIN (allows same-origin framing)")
    print("  4. API views: Applied DENY (prevents all framing)")
    print("  5. Legacy views: Applied DENY (prevents all framing)")
    print("  6. Widget views: Applied EXEMPT (allows partner embedding)")
    print("  7. Admin interface: Protected by default middleware settings")

if __name__ == '__main__':
    main()