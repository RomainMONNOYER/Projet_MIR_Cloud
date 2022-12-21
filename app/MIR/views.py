import os.path

import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader


from .forms import ImageForm, SearchForm
from .models import ImageRequests
from .utils import extractReqFeatures, getkVoisins2_files, Compute_RP, extractReqFeatures2, getkVoisins2_files222222
import time


def index(request, *args, **kwargs):
    latest_question_list = "This is my question"
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
            vec, descriptor = extractReqFeatures(image.image.url, algo_choice=form.instance)
            tmp = getkVoisins2_files(vec, form.instance.top, descriptor, form.instance.distance)
            end = time.time()
            voisins = [tmp[i][1] for i in range(len(tmp))]
            noms_voisins = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[0]) for voisin in voisins]
            noms_voisins2 = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[1]) for voisin in voisins]
            graph1,mean_r, mean_p = Compute_RP(form.instance.top,
                                image.classification,
                                noms_voisins,
                                descriptor,
                                form.instance.distance)
            graph2,_,_ =Compute_RP(form.instance.top,
                       image.subclassification,
                       noms_voisins2,
                       descriptor,
                       form.instance.distance)
            graph3,_,_ = Compute_RP(form.instance.top,
                                  image.subclassification,
                                  noms_voisins2,
                                  descriptor,
                                  form.instance.distance,
                                  rp_process = 'Mrp')
            return render(request, 'search.html', {'pk': image.image,
                                                   'class': ImageRequests.ClassChoices(image.classification).name,
                                                   'subclass': ImageRequests.SubClassChoices(image.classification).name,
                                                   'form': form,
                                                   'voisins':voisins,
                                                   'time': round(end-start,2),
                                                   'MeanR':mean_r,
                                                   'MeanP':mean_p,
                                                   'graph': graph1,
                                                   'graph2':graph2,
                                                   'graph3':graph3})

    else:
        form = SearchForm()
    return render(request, 'search.html', {'pk': image.image, 'form': form})
def image_search2(request, pk):
    image = ImageRequests.objects.filter(id = pk).first()
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            vec1, descriptor1 = extractReqFeatures2(image.image.url, algo_choice=form.instance.descriptor1)
            vec2, descriptor2 = extractReqFeatures2(image.image.url, algo_choice=form.instance.descriptor2)
            start = time.time()
            tmp = getkVoisins2_files222222(np.concatenate([vec1, vec2]), form.instance.top, descriptor1, descriptor2, form.instance.distance)
            end = time.time()
            voisins = [tmp[i][1] for i in range(len(tmp))]
            noms_voisins = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[0]) for voisin in voisins]
            graph1,mean_r, mean_p = Compute_RP(form.instance.top,
                                               image.classification,
                                               noms_voisins,
                                               f"{descriptor1} + {descriptor2}",
                                               form.instance.distance)
            return render(request, 'search.html', {'pk': image.image,
                                                   'class': ImageRequests.ClassChoices(image.classification).name,
                                                   'subclass': ImageRequests.SubClassChoices(image.classification).name,
                                                   'form': form,
                                                   'voisins':voisins,
                                                   'time': round(end-start,2),
                                                   'MeanR':mean_r,
                                                   'MeanP':mean_p,
                                                   'graph': graph1,})
    else:
        form = SearchForm()
    return render(request, 'search.html', {'pk': image.image, 'form': form})
