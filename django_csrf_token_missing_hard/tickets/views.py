from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse


@require_http_methods(["GET", "POST"])
def create_ticket(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description')
        priority = request.POST.get('priority')
        
        if title and description and priority:
            # Create ticket logic here
            # ticket = Ticket.objects.create(
            #     title=title,
            #     description=description,
            #     priority=priority
            # )
            
            messages.success(request, 'Ticket created successfully!')
            return redirect('create_ticket')
        else:
            messages.error(request, 'All fields are required.')
    
    return render(request, 'tickets/create_ticket.html')