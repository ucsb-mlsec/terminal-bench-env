from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods

@require_http_methods(['GET', 'POST'])
def contact_view(request):
    if request.method == 'GET':
        return render(request, 'contact.html')
    if request.method == 'POST':
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        message = request.POST.get('message', '')
        return HttpResponse(f'Thank you {name}! Your message has been received.')