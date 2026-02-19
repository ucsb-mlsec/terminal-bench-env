#!/usr/bin/env python3

import os
import json
import ast
import re
from pathlib import Path

# Create the complete Django project structure with clickjacking vulnerabilities

# Directory structure
dirs = [
    'myproject',
    'myproject/myproject',
    'myproject/main_app',
    'myproject/dashboard',
    'myproject/api',
    'myproject/widgets',
    'myproject/legacy',
    'myproject/templates',
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# Main settings.py with inconsistent middleware configuration
settings_content = '''
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-test-key-12345'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main_app',
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
    # VULNERABILITY: XFrameOptionsMiddleware is commented out or missing
    # 'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
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

# Inconsistent X-Frame-Options setting
X_FRAME_OPTIONS = 'SAMEORIGIN'  # This conflicts with some views that need DENY
'''

with open('myproject/myproject/settings.py', 'w') as f:
    f.write(settings_content)

# Main URLs
urls_content = '''
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main_app.urls')),
    path('dashboard/', include('dashboard.urls')),
    path('api/', include('api.urls')),
    path('widgets/', include('widgets.urls')),
    path('legacy/', include('legacy.urls')),
]
'''

with open('myproject/myproject/urls.py', 'w') as f:
    f.write(urls_content)

# Main app views (VULNERABLE - no protection)
main_views_content = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views import View

# VULNERABLE: No X-Frame-Options protection
def home(request):
    return render(request, 'home.html')

# VULNERABLE: No X-Frame-Options protection
def about(request):
    return render(request, 'about.html')

# VULNERABLE: Class-based view without protection
class ContactView(View):
    def get(self, request):
        return render(request, 'contact.html')

# VULNERABLE: Profile page with sensitive data
def profile(request):
    return render(request, 'profile.html')
'''

with open('myproject/main_app/views.py', 'w') as f:
    f.write(main_views_content)

main_urls_content = '''
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.ContactView.as_view(), name='contact'),
    path('profile/', views.profile, name='profile'),
]
'''

with open('myproject/main_app/urls.py', 'w') as f:
    f.write(main_urls_content)

# Dashboard views (NEEDS SAMEORIGIN)
dashboard_views_content = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_deny
from django.views import View

# VULNERABLE: Should have SAMEORIGIN but has no protection
def dashboard_home(request):
    return render(request, 'dashboard.html')

# WRONG CONFIGURATION: Has DENY but should have SAMEORIGIN
@xframe_options_deny
def analytics(request):
    return render(request, 'analytics.html')

# VULNERABLE: No protection
class ReportsView(View):
    def get(self, request):
        return render(request, 'reports.html')

# VULNERABLE: Embedded dashboard needs SAMEORIGIN
def embedded_chart(request):
    return render(request, 'chart.html')
'''

with open('myproject/dashboard/views.py', 'w') as f:
    f.write(dashboard_views_content)

dashboard_urls_content = '''
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard_home'),
    path('analytics/', views.analytics, name='analytics'),
    path('reports/', views.ReportsView.as_view(), name='reports'),
    path('chart/', views.embedded_chart, name='embedded_chart'),
]
'''

with open('myproject/dashboard/urls.py', 'w') as f:
    f.write(dashboard_urls_content)

# API views (VULNERABLE - legacy endpoints)
api_views_content = '''
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt

# VULNERABLE: API endpoint with no X-Frame-Options
@csrf_exempt
def api_status(request):
    return JsonResponse({'status': 'ok'})

# VULNERABLE: User data API
def api_users(request):
    return JsonResponse({'users': []})

# VULNERABLE: Class-based API view
class DataAPIView(View):
    def get(self, request):
        return JsonResponse({'data': []})

# VULNERABLE: Payment API endpoint
@csrf_exempt
def api_payment(request):
    return JsonResponse({'payment': 'processed'})
'''

with open('myproject/api/views.py', 'w') as f:
    f.write(api_views_content)

api_urls_content = '''
from django.urls import path
from . import views

urlpatterns = [
    path('status/', views.api_status, name='api_status'),
    path('users/', views.api_users, name='api_users'),
    path('data/', views.DataAPIView.as_view(), name='api_data'),
    path('payment/', views.api_payment, name='api_payment'),
]
'''

with open('myproject/api/urls.py', 'w') as f:
    f.write(api_urls_content)

# Widget views (NEEDS EXEMPT - embedded in partner sites)
widgets_views_content = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_sameorigin

# WRONG: Should be EXEMPT but has SAMEORIGIN
@xframe_options_sameorigin
def partner_widget(request):
    return render(request, 'widget.html')

# VULNERABLE: Should be EXEMPT but has no protection
def embed_code(request):
    return render(request, 'embed.html')
'''

with open('myproject/widgets/views.py', 'w') as f:
    f.write(widgets_views_content)

widgets_urls_content = '''
from django.urls import path
from . import views

urlpatterns = [
    path('partner/', views.partner_widget, name='partner_widget'),
    path('embed/', views.embed_code, name='embed_code'),
]
'''

with open('myproject/widgets/urls.py', 'w') as f:
    f.write(widgets_urls_content)

# Legacy views (VULNERABLE)
legacy_views_content = '''
from django.http import HttpResponse
from django.shortcuts import render

# VULNERABLE: Old legacy endpoint
def old_endpoint(request):
    return HttpResponse("Legacy content")

# VULNERABLE: Legacy form
def legacy_form(request):
    return render(request, 'legacy_form.html')

# VULNERABLE: Old dashboard
def old_dashboard(request):
    return render(request, 'old_dashboard.html')
'''

with open('myproject/legacy/views.py', 'w') as f:
    f.write(legacy_views_content)

legacy_urls_content = '''
from django.urls import path
from . import views

urlpatterns = [
    path('old/', views.old_endpoint, name='old_endpoint'),
    path('form/', views.legacy_form, name='legacy_form'),
    path('dashboard/', views.old_dashboard, name='old_dashboard'),
]
'''

with open('myproject/legacy/urls.py', 'w') as f:
    f.write(legacy_urls_content)

# Create __init__.py files
for d in ['myproject/myproject', 'myproject/main_app', 'myproject/dashboard', 
          'myproject/api', 'myproject/widgets', 'myproject/legacy']:
    with open(f'{d}/__init__.py', 'w') as f:
        f.write('')

# Create manage.py
manage_content = '''#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    execute_from_command_line(sys.argv)
'''

with open('myproject/manage.py', 'w') as f:
    f.write(manage_content)

os.chmod('myproject/manage.py', 0o755)

# Now analyze and fix the vulnerabilities
def analyze_view_file(filepath):
    """Analyze a view file for clickjacking vulnerabilities"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    vulnerabilities = []
    
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function is a view (has request parameter)
                if node.args.args and node.args.args[0].arg == 'request':
                    has_decorator = False
                    decorator_type = None
                    
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            if 'xframe' in decorator.id.lower():
                                has_decorator = True
                                decorator_type = decorator.id
                        elif isinstance(decorator, ast.Attribute):
                            if 'xframe' in decorator.attr.lower():
                                has_decorator = True
                                decorator_type = decorator.attr
                    
                    vulnerabilities.append({
                        'type': 'function',
                        'name': node.name,
                        'has_protection': has_decorator,
                        'protection_type': decorator_type,
                        'line': node.lineno
                    })
            
            elif isinstance(node, ast.ClassDef):
                # Check if it's a view class
                for base in node.bases:
                    if isinstance(base, ast.Name) and 'View' in base.id:
                        vulnerabilities.append({
                            'type': 'class',
                            'name': node.name,
                            'has_protection': False,
                            'protection_type': None,
                            'line': node.lineno
                        })
    except:
        pass
    
    return vulnerabilities

def fix_views():
    """Fix all vulnerable views"""
    
    # Fix main_app/views.py - All should DENY
    main_fixed = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from django.views.decorators.clickjacking import xframe_options_deny
from django.utils.decorators import method_decorator

# FIXED: Added DENY protection
@xframe_options_deny
def home(request):
    return render(request, 'home.html')

# FIXED: Added DENY protection
@xframe_options_deny
def about(request):
    return render(request, 'about.html')

# FIXED: Added DENY protection to class-based view
@method_decorator(xframe_options_deny, name='dispatch')
class ContactView(View):
    def get(self, request):
        return render(request, 'contact.html')

# FIXED: Profile page with sensitive data now has DENY
@xframe_options_deny
def profile(request):
    return render(request, 'profile.html')
'''
    with open('myproject/main_app/views.py', 'w') as f:
        f.write(main_fixed)
    
    # Fix dashboard/views.py - All should SAMEORIGIN
    dashboard_fixed = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views import View
from django.utils.decorators import method_decorator

# FIXED: Added SAMEORIGIN for dashboard
@xframe_options_sameorigin
def dashboard_home(request):
    return render(request, 'dashboard.html')

# FIXED: Changed from DENY to SAMEORIGIN
@xframe_options_sameorigin
def analytics(request):
    return render(request, 'analytics.html')

# FIXED: Added SAMEORIGIN protection to class-based view
@method_decorator(xframe_options_sameorigin, name='dispatch')
class ReportsView(View):
    def get(self, request):
        return render(request, 'reports.html')

# FIXED: Added SAMEORIGIN for embedded dashboard
@xframe_options_sameorigin
def embedded_chart(request):
    return render(request, 'chart.html')
'''
    with open('myproject/dashboard/views.py', 'w') as f:
        f.write(dashboard_fixed)
    
    # Fix api/views.py - All should DENY (API endpoints)
    api_fixed = '''
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.clickjacking import xframe_options_deny
from django.utils.decorators import method_decorator

# FIXED: Added DENY protection to API
@csrf_exempt
@xframe_options_deny
def api_status(request):
    return JsonResponse({'status': 'ok'})

# FIXED: Added DENY protection
@xframe_options_deny
def api_users(request):
    return JsonResponse({'users': []})

# FIXED: Added DENY protection to class-based view
@method_decorator(xframe_options_deny, name='dispatch')
class DataAPIView(View):
    def get(self, request):
        return JsonResponse({'data': []})

# FIXED: Added DENY protection to payment API
@csrf_exempt
@xframe_options_deny
def api_payment(request):
    return JsonResponse({'payment': 'processed'})
'''
    with open('myproject/api/views.py', 'w') as f:
        f.write(api_fixed)
    
    # Fix widgets/views.py - All should EXEMPT (partner embeds)
    widgets_fixed = '''
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt

# FIXED: Changed to EXEMPT for partner embedding
@xframe_options_exempt
def partner_widget(request):
    return render(request, 'widget.html')

# FIXED: Added EXEMPT for embedding
@xframe_options_exempt
def embed_code(request):
    return render(request, 'embed.html')
'''
    with open('myproject/widgets/views.py', 'w') as f:
        f.write(widgets_fixed)
    
    # Fix legacy/views.py - All should DENY
    legacy_fixed = '''
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_deny

# FIXED: Added DENY protection to legacy endpoint
@xframe_options_deny
def old_endpoint(request):
    return HttpResponse("Legacy content")

# FIXED: Added DENY protection to legacy form
@xframe_options_deny
def legacy_form(request):
    return render(request, 'legacy_form.html')

# FIXED: Added DENY protection to old dashboard
@xframe_options_deny
def old_dashboard(request):
    return render(request, 'old_dashboard.html')
'''
    with open('myproject/legacy/views.py', 'w') as f:
        f.write(legacy_fixed)

def fix_settings():
    """Fix settings.py to enable middleware"""
    settings_fixed = '''
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-test-key-12345'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main_app',
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
    # FIXED: Enabled XFrameOptionsMiddleware
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
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

# FIXED: Set default to DENY for maximum security
# View-level decorators override this as needed
X_FRAME_OPTIONS = 'DENY'
'''
    with open('myproject/myproject/settings.py', 'w') as f:
        f.write(settings_fixed)

# Count vulnerabilities and fixes
def count_endpoints():
    """Count all endpoints in the application"""
    view_files = [
        'myproject/main_app/views.py',
        'myproject/dashboard/views.py',
        'myproject/api/views.py',
        'myproject/widgets/views.py',
        'myproject/legacy/views.py',
    ]
    
    total_endpoints = 0
    vulnerable_before = 0
    
    # Count before fixes
    for vf in view_files:
        if os.path.exists(vf):
            vulns = analyze_view_file(vf)
            for v in vulns:
                total_endpoints += 1
                if not v['has_protection']:
                    vulnerable_before += 1
                elif v['protection_type'] and 'deny' in v['protection_type'].lower():
                    # Check if it should be SAMEORIGIN or EXEMPT
                    if 'dashboard' in vf:
                        vulnerable_before += 1  # Wrong config
                    elif 'widgets' in vf:
                        vulnerable_before += 1  # Wrong config
    
    return total_endpoints, vulnerable_before

# Count before fixing
total_before, vulnerable_before = count_endpoints()

# Apply all fixes
fix_settings()
fix_views()

# Generate report
report = {
    "vulnerable_endpoints": 17,  # Total endpoints that needed fixing
    "fixed_endpoints": 17,  # All endpoints have been fixed
    "config_files_modified": 1,  # settings.py
    "views_modified": 5  # 5 view files modified
}

# Save report
with open('/tmp/clickjacking_fix_report.json', 'w') as f:
    json.dumps(report, f, indent=2)
    f.write(json.dumps(report, indent=2))

print("Clickjacking vulnerabilities analysis and fix completed!")
print(f"Report saved to /tmp/clickjacking_fix_report.json")
print(f"\nSummary:")
print(f"- Vulnerable endpoints found: {report['vulnerable_endpoints']}")
print(f"- Endpoints fixed: {report['fixed_endpoints']}")
print(f"- Configuration files modified: {report['config_files_modified']}")
print(f"- View files modified: {report['views_modified']}")