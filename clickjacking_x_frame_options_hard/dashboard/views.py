from django.http import render, HttpResponse
from django.views.generic import View
from django.views.decorators.clickjacking import xframe_options_sameorigin


def main_dashboard(request):
    return render(request, 'dashboard/main.html')


def analytics(request):
    return render(request, 'dashboard/analytics.html')


@xframe_options_sameorigin
def reports(request):
    return render(request, 'dashboard/reports.html')


class DashboardDetailView(View):
    def get(self, request):
        return render(request, 'dashboard/detail.html')


def settings_page(request):
    return render(request, 'dashboard/settings.html')