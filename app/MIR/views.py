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
    try:
        if request.method == 'POST':
            form = SearchForm(request.POST)
            if form.is_valid():
                if (
                    form.instance.descriptor1
                    in (
                        DescriptorRequests.DescriptorChoices.SIFT,
                        DescriptorRequests.DescriptorChoices.ORB,
                    )
                ) and form.instance.distance not in (
                    DescriptorRequests.DistanceChoices.FLANN,
                    DescriptorRequests.DistanceChoices.BRUTE_FORCE,
                ):
                    return render(request, 'search.html', {'pk': image.image, 'class': ImageRequests.ClassChoices(image.classification).name,'form': form, '2_descriptors': False,'oneDescriptor':True, 'Error': 'Descriptor and distance mismatch'})
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
                graph2,_,_, _,_= Compute_RP(top,
                                            image.classification,
                                            noms_voisins,
                                            descriptor,
                                            form.instance.distance, r=10,
                                            rp_process = 'Mrp')
                if form.instance.top == DescriptorRequests.TopChoices.TOP_MAX:
                    top = get_top(image.subclassification)
                else:
                    top=form.instance.top
                sub_noms_voisins = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[1]) for voisin in voisins]
                sub_graph1,sub_mean_r, sub_mean_p, sub_r_precision, sub_f_mesure = Compute_RP(top,
                                                                          int(image.subclassification%10),
                                                                          sub_noms_voisins,
                                                                          descriptor,
                                                                          form.instance.distance,r=form.instance.R_precision)
                sub_graph2,_,_, _,_= Compute_RP(top,
                                            int(image.subclassification%10),
                                            sub_noms_voisins,
                                            descriptor,
                                            form.instance.distance, r=10,
                                            rp_process = 'Mrp')
                return render(request, 'search.html', {'pk': image.image,
                                                       'class': ImageRequests.ClassChoices(image.classification).name,
                                                       'subclass': ImageRequests.SubClassChoices(image.subclassification).name,
                                                       'form': form,
                                                       '2_descriptors': False,
                                                       'voisins':tmp,
                                                       'time': round(end-start,2),
                                                       'MeanR':mean_r,
                                                       'MeanP':mean_p,
                                                       'Rprecision': r_precision,
                                                       'Fmesure': f_mesure,
                                                       'graph': graph1,
                                                       'graph3':graph2,

                                                       'subMeanR':sub_mean_r,
                                                       'subMeanP':sub_mean_p,
                                                       'subRprecision': sub_r_precision,
                                                       'subFmesure': sub_f_mesure,
                                                       'subgraph': sub_graph1,
                                                       'subgraph3':sub_graph2,
                                                       'oneDescriptor':True})

        else:
            form = SearchForm()
            return render(request, 'search.html', {'pk': image.image, 'class': ImageRequests.ClassChoices(image.classification).name,
                                                   'subclass': ImageRequests.SubClassChoices(image.subclassification).name,'form': form, '2_descriptors': False,'oneDescriptor':True})
    except:
        return render(request, 'search.html', {'pk': image.image, 'class': ImageRequests.ClassChoices(image.classification).name,
                                               'subclass': ImageRequests.SubClassChoices(image.subclassification).name,'form': form, '2_descriptors': False,'oneDescriptor':True, 'Error': 'Something wrong happened'})


@login_required()
def image_search2(request, pk):
    image = ImageRequests.objects.filter(id = pk).first()
    try:
        if request.method == 'POST':
            form = SearchForm(request.POST)
            if form.is_valid():
                if (
                        form.instance.descriptor1
                        in (
                                DescriptorRequests.DescriptorChoices.SIFT,
                                DescriptorRequests.DescriptorChoices.ORB,
                        )
                        or form.instance.descriptor2
                        in (
                                DescriptorRequests.DescriptorChoices.SIFT,
                                DescriptorRequests.DescriptorChoices.ORB,
                        )
                ) and form.instance.distance not in (
                        DescriptorRequests.DistanceChoices.FLANN,
                        DescriptorRequests.DistanceChoices.BRUTE_FORCE,
                ):
                    return render(request, 'search.html', {'pk': image.image, 'class': ImageRequests.ClassChoices(image.classification).name,'form': form, '2_descriptors': False,'oneDescriptor':True, 'Error': 'Descriptor and distance mismatch'})

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

                if form.instance.top == DescriptorRequests.TopChoices.TOP_MAX:
                    top = get_top(image.subclassification)
                else:
                    top=form.instance.top
                sub_noms_voisins = [int(os.path.splitext(os.path.basename(voisin))[0].split("_")[1]) for voisin in voisins]
                sub_graph1,sub_mean_r, sub_mean_p, sub_r_precision, sub_f_mesure = Compute_RP(top,
                                                                                              int(image.subclassification%10),
                                                                                              sub_noms_voisins,
                                                                                              f"{descriptor1} + {descriptor2}",
                                                                                              form.instance.distance,r=form.instance.R_precision)
                sub_graph2,_,_, _,_= Compute_RP(top,
                                                int(image.subclassification%10),
                                                sub_noms_voisins,
                                                f"{descriptor1} + {descriptor2}",
                                                form.instance.distance, r=10,
                                                rp_process = 'Mrp')


                return render(request, 'search.html', {'pk': image.image,
                                                       'class': ImageRequests.ClassChoices(image.classification).name,
                                                       'subclass': ImageRequests.SubClassChoices(image.subclassification).name,
                                                       'form': form,
                                                       '2_descriptors': True,
                                                       'voisins':tmp,
                                                       'time': round(end-start,2),
                                                       'MeanR':mean_r,
                                                       'MeanP':mean_p,
                                                       'Rprecision': r_precision,
                                                       'Fmesure': f_mesure,
                                                       'graph': graph1,
                                                       'graph3':graph3,

                                                       'subMeanR':sub_mean_r,
                                                       'subMeanP':sub_mean_p,
                                                       'subRprecision': sub_r_precision,
                                                       'subFmesure': sub_f_mesure,
                                                       'subgraph': sub_graph1,
                                                       'subgraph3':sub_graph2,


                                                       'oneDescriptor':False})
        else:
            form = SearchForm()
            return render(request, 'search.html', {'pk': image.image,'class': ImageRequests.ClassChoices(image.classification).name,
                                                   'subclass': ImageRequests.SubClassChoices(image.subclassification).name, 'form': form, 'oneDescriptor':False})
    except Exception as e:
        print(e)

        return render(request, 'search.html', {'pk': image.image, 'class': ImageRequests.ClassChoices(image.classification).name,
                                               'subclass': ImageRequests.SubClassChoices(image.subclassification).name,'form': form, '2_descriptors': True,'oneDescriptor':False, 'Error': "Something wrong happened"})


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
