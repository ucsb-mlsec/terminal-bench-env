from django.http import HttpResponse, render
from django.views.generic import View
from django.views.decorators.clickjacking import xframe_options_exempt

@xframe_options_exempt
def embeddable_widget(request):
    # Must be embeddable on partner sites
    return render(request, 'widgets/embed.html')

def widget_config(request):
    return render(request, 'widgets/config.html')

class WidgetAPIView(View):
    def get(self, request):
        return HttpResponse('Widget API')