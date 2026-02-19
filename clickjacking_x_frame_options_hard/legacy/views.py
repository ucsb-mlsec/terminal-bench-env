# Legacy views - TODO: Add security
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.generic import View


def old_endpoint(request):
    return HttpResponse('Legacy endpoint')


def legacy_api(request):
    return JsonResponse({'legacy': True})


def deprecated_view(request):
    return render(request, 'legacy/old.html')


class LegacyDataView(View):
    def get(self, request):
        return JsonResponse({'old_data': 'value'})