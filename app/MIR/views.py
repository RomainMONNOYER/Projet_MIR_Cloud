import os.path

from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import ImageForm, SearchForm
from .models import ImageRequests
from .utils import extractReqFeatures, getkVoisins, getkVoisins2_files


def index(request, *args, **kwargs):
    latest_question_list = "This is my question"
    print(f"Form: {request}")
    template = loader.get_template('test.html')
    context = {
        'latest_question_list': [latest_question_list],
    }
    return HttpResponse(template.render(context, request))


def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':

        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            return redirect(f"/search/{img_obj.id}")

    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})

def image_search(request, pk):
    image = ImageRequests.objects.filter(id = pk).first()
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            form.save()
            vec, json_data = extractReqFeatures(image.image.url, algo_choice=form.instance)
            tmp = getkVoisins2_files(vec, form.instance.top, json_data)
            voisins = [tmp[i][1] for i in range(len(tmp))]
            print(tmp)
            return render(request, 'search.html', {'pk': image.image, 'form': form, 'voisins':voisins})

    else:
        form = SearchForm()
    return render(request, 'search.html', {'pk': image.image, 'form': form})