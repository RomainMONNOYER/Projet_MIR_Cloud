import os.path

import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader


from .forms import ImageForm, SearchForm
from .models import ImageRequests
from .utils import extractReqFeatures, getkVoisins2_files, Compute_RP
import time


def index(request, *args, **kwargs):
    latest_question_list = "This is my question"
    print(f"Form: {request}")
    template = loader.get_template('test.html')
    context = {
        'latest_question_list': [latest_question_list],
    }
    return HttpResponse(template.render(context, request))


def image_upload_view(request):
    if request.method == 'POST':

        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
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
            start = time.time()
            vec, json_data = extractReqFeatures(image.image.url, algo_choice=form.instance)
            tmp = getkVoisins2_files(vec, form.instance.top, json_data, form.instance.distance)
            end = time.time()
            voisins = [tmp[i][1] for i in range(len(tmp))]
            noms_voisins = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[0]) for voisin in voisins]
            noms_voisins2 = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[1]) for voisin in voisins]
            Compute_RP(form.instance.top, image.classification, noms_voisins)
            return render(request, 'search.html', {'pk': image.image,
                                                   'form': form,
                                                   'voisins':voisins,
                                                   'time': round(end-start,2),
                                                   'graph': Compute_RP(form.instance.top,
                                                                       image.classification,
                                                                       noms_voisins),
                                                   'graph2':Compute_RP(form.instance.top,
                                                                       image.subclassification,
                                                                       noms_voisins2)})
    else:
        form = SearchForm()
    return render(request, 'search.html', {'pk': image.image, 'form': form})


import matplotlib.pyplot as plt
from io import StringIO

def return_graph():

    x = np.arange(0,np.pi*3,.1)
    y = np.sin(x)

    fig = plt.figure()
    plt.plot(x,y)

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data