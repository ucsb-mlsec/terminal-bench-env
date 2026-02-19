from django.http import JsonResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.decorators.clickjacking import xframe_options_deny


def data_endpoint(request):
    return JsonResponse({'data': 'value'})


@csrf_exempt
@xframe_options_deny
def public_api(request):
    return JsonResponse({'status': 'ok'})


def user_info(request):
    return JsonResponse({'user': 'info'})


@method_decorator(csrf_exempt, name='dispatch')
class DataView(View):
    def get(self, request):
        return JsonResponse({'items': []})


@csrf_exempt
def webhooks(request):
    return JsonResponse({'received': True})


def stats(request):
    return JsonResponse({'stats': {}})