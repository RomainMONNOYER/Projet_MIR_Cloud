from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import ImageForm, SearchForm
from .models import ImageRequests
from .utils import extractReqFeatures


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
        print(f"Post request datas: {request.POST}")
        print(f"Request: {request}")
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            print(f"Form instance: {form.instance.id}")
            # Get the current instance object to display in the template
            img_obj = form.instance
            return redirect(f"/MIR/search/{img_obj.id}")

    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})

def image_search(request, pk):
    image = ImageRequests.objects.filter(id = pk).first()
    if request.method == 'POST':
        form = SearchForm(request.POST)
        extractReqFeatures(image.image.url)
        if form.is_valid():
            return render(request, 'search.html', {'pk': image.image, 'form': form})
    else:
        form = SearchForm()
    return render(request, 'search.html', {'pk': image.image, 'form': form})