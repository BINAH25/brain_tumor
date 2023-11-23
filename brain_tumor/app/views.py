from django.shortcuts import render, redirect
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from . models import File 
from .utils import *

# Create your views here.

def home(request):
    return render(request, 'index.html')


@csrf_exempt
def predict_view(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file = File.objects.create(file=uploaded_file)
        url = file.file.url
        url = str(settings.BASE_DIR)+url

        # Get result and class name
        value = get_result(url)
        result = result_interpretation(value)
        
        print('numeric value:', value)
        print(result)

        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request'})