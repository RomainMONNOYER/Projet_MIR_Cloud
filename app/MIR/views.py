import os.path

import numpy as np
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader


from .forms import ImageForm, SearchForm
from .models import ImageRequests, DescriptorRequests
from .utils import extractReqFeatures, getkVoisins2_files, Compute_RP, extractReqFeatures, getkVoisins2_files222222, \
    get_top
import time
@login_required()
def home(request, *args, **kwargs):
    return render(request, 'manual.html')

@login_required()
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
def image_upload_view2(request):
    if request.method == 'POST':

        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            return redirect(f"/search-2descriptors/{img_obj.id}")

    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})

@login_required()
def image_search(request, pk):
    image = ImageRequests.objects.filter(id = pk).first()
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            form.save()
            if form.instance.top == DescriptorRequests.TopChoices.TOP_MAX:
                top = get_top(image.classification)
            else:
                top=form.instance.top
            start = time.time()
            vec, descriptor = extractReqFeatures(image.image.url, algo_choice=form.instance.descriptor1)
            tmp = getkVoisins2_files(vec, top, descriptor, form.instance.distance)
            end = time.time()
            voisins = [tmp[i][1] for i in range(len(tmp))]
            noms_voisins = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[0]) for voisin in voisins]
            graph1,mean_r, mean_p, r_precision, f_mesure = Compute_RP(top,
                                image.classification,
                                noms_voisins,
                                descriptor,
                                form.instance.distance,r=form.instance.R_precision)
            graph3,_,_, _,_= Compute_RP(top,
                                  image.classification,
                                  noms_voisins,
                                  descriptor,
                                  form.instance.distance, r=10,
                                  rp_process = 'Mrp')
            return render(request, 'search.html', {'pk': image.image,
                                                   'class': ImageRequests.ClassChoices(image.classification).name,
                                                   'form': form,
                                                   '2_descriptors': False,
                                                   'voisins':voisins,
                                                   'time': round(end-start,2),
                                                   'MeanR':mean_r,
                                                   'MeanP':mean_p,
                                                   'Rprecision': r_precision,
                                                   'Fmesure': f_mesure,
                                                   'graph': graph1,
                                                   'graph3':graph3,
                                                   'oneDescriptor':True})

    else:
        form = SearchForm()
    return render(request, 'search.html', {'pk': image.image, 'class': ImageRequests.ClassChoices(image.classification).name,'form': form, '2_descriptors': False,'oneDescriptor':True})

@login_required()
def image_search2(request, pk):
    image = ImageRequests.objects.filter(id = pk).first()
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            if form.instance.top == DescriptorRequests.TopChoices.TOP_MAX:
                top = get_top(image.classification)
            else:
                top=form.instance.top
            vec1, descriptor1 = extractReqFeatures(image.image.url, algo_choice=form.instance.descriptor1)
            vec2, descriptor2 = extractReqFeatures(image.image.url, algo_choice=form.instance.descriptor2)
            start = time.time()
            tmp = getkVoisins2_files222222(np.concatenate([vec1, vec2]), top, descriptor1, descriptor2, form.instance.distance)
            end = time.time()
            voisins = [tmp[i][1] for i in range(len(tmp))]
            noms_voisins = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[0]) for voisin in voisins]
            graph1,mean_r, mean_p,r_precision, f_mesure = Compute_RP(top,
                                               image.classification,
                                               noms_voisins,
                                               f"{descriptor1} + {descriptor2}",
                                               form.instance.distance, r=form.instance.R_precision)
            graph3,_,_,_,_ = Compute_RP(top,
                                    image.classification,
                                    noms_voisins,
                                    f"{descriptor1} + {descriptor2}",
                                    form.instance.distance, r=10,
                                    rp_process = 'Mrp')
            return render(request, 'search.html', {'pk': image.image,
                                                   'class': ImageRequests.ClassChoices(image.classification).name,
                                                   'form': form,
                                                   '2_descriptors': True,
                                                   'voisins':voisins,
                                                   'time': round(end-start,2),
                                                   'MeanR':mean_r,
                                                   'MeanP':mean_p,
                                                   'Rprecision': r_precision,
                                                   'Fmesure': f_mesure,
                                                   'graph': graph1,
                                                   'graph3':graph3,
                                                   'oneDescriptor':False})
    else:
        form = SearchForm()
    return render(request, 'search.html', {'pk': image.image, 'form': form, 'oneDescriptor':False})

@login_required()
def image_from_db(request):
    db_images = ImageRequests.objects.filter(is_database_img=True)
    images = [(image.id, os.path.join(settings.MEDIA_URL, str(image.image)), image.title, ImageRequests.ClassChoices(image.classification).name) for image in db_images]
    return render(request, 'db.html', {'images':images,
                                       'title':'database'})

@login_required
def image_history(request):
    db_images = ImageRequests.objects.all().order_by('-date_upload')
    images = [(image.id, os.path.join(settings.MEDIA_URL, str(image.image)), image.title, ImageRequests.ClassChoices(image.classification).name) for image in db_images]
    return render(request, 'db.html', {'images':images,
                                       'title': 'history'})
