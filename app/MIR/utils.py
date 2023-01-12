# Defintion de toute les fonctions à appeller dans l'interface
import json
import operator
import os
from io import StringIO

import cv2
import numpy as np
from django.conf import settings
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import io, color, img_as_ubyte
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern

import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from .distances import euclidean, chiSquareDistance, bhatta, bruteForceMatching, flann
from .enums import TopClassChoices
from .models import DescriptorRequests, ImageRequests


def BGR(img):
    histB = cv2.calcHist([img], [0], None, [256], [0, 256])
    histG = cv2.calcHist([img], [1], None, [256], [0, 256])
    histR = cv2.calcHist([img], [2], None, [256], [0, 256])
    return np.concatenate((histB, np.concatenate((histG, histR), axis=None)), axis=None)


def HSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    histH = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    histS = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    histV = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    return np.concatenate((histH, np.concatenate((histS, histV), axis=None)), axis=None)


def SIFT(img1):
    img = cv2.resize(img1, (128*2, 64*2))
    sift = cv2.SIFT_create()
    _ , vect_features = sift.detectAndCompute(img,None)
    return vect_features
def ORB(img):
    orb = cv2.ORB_create()
    _, vect_features = orb.detectAndCompute(img, None)
    return vect_features


def GLCM(img):
    distances = [1, -1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img_as_ubyte(gray)
    glcmMatrix = greycomatrix(gray, distances=distances, angles=angles, normed=True)
    glcmProperties1 = greycoprops(glcmMatrix, 'contrast').ravel()
    glcmProperties2 = greycoprops(glcmMatrix, 'dissimilarity').ravel()
    glcmProperties3 = greycoprops(glcmMatrix, 'homogeneity').ravel()
    glcmProperties4 = greycoprops(glcmMatrix, 'energy').ravel()
    glcmProperties5 = greycoprops(glcmMatrix, 'correlation').ravel()
    glcmProperties6 = greycoprops(glcmMatrix, 'ASM').ravel()
    return np.array(
        [glcmProperties1, glcmProperties2, glcmProperties3, glcmProperties4, glcmProperties5, glcmProperties6]).ravel()


def LBP(img):
    points = 8
    radius = 1
    method = 'default'
    subSize = (70, 70)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (350, 350))
    fullLBPmatrix = local_binary_pattern(img, points, radius, method)
    histograms = []
    for k in range(int(fullLBPmatrix.shape[0] / subSize[0])):
        for j in range(int(fullLBPmatrix.shape[1] / subSize[1])):
            subVector = fullLBPmatrix[k * subSize[0]:(k + 1) * subSize[0],
                        j * subSize[1]:(j + 1) * subSize[1]].ravel()
            subHist, edges = np.histogram(subVector, bins=int(2 ** points), range=(0, 2 ** points))
            histograms = np.concatenate((histograms, subHist), axis=None)
    return histograms


def HOG(img):
    cellSize = (25, 25)
    blockSize = (50, 50)
    blockStride = (25, 25)
    nBins = 9
    winSize = (350, 350)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, winSize)
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins)
    return hog.compute(image)


def VGG16(fileName):
    model = settings.VGG16
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image)  # predict the probability
    return np.array(feature[0])
def RESNET101(fileName):
    model = settings.RESNET101
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image)  # predict the probability
    return np.array(feature[0])
def RESNET101_1(fileName):
    model = settings.RESNET101
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = Model(inputs=settings.RESNET101.input, outputs=settings.RESNET101.layers[-2].output).predict(image)# predict the probability
    return np.array(feature[0])
def RESNET50(fileName):
    model = settings.RESNET50
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image)  # predict the probability
    return np.array(feature[0])

def RESNET50_1(fileName):
    model = settings.RESNET50
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = Model(inputs=settings.RESNET50.input, outputs=settings.RESNET50.layers[-2].output).predict(image)# predict the probability
    return np.array(feature[0])

def VGG16_1(fileName):
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature =  Model(inputs=settings.VGG16.input, outputs=settings.VGG16.layers[-2].output).predict(image)  # predict the probability
    return np.array(feature[0])
def MOBILENET(fileName):
    model = settings.MOBILENET
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image)  # predict the probability
    return np.array(feature[0])
def XCEPTION(fileName):
    model = settings.XCEPTION
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image)  # predict the probability
    return np.array(feature[0])

def getkVoisins2_files(vec_descriptor, top, descriptor_folder, distance_choice):
    ldistances = []
    for path, subdir, files in os.walk(os.path.join(settings.MEDIA_ROOT, descriptor_folder)):
        for f in files:
            p = os.path.join(path, f)
            if os.path.getsize(p) == 0:
                continue
            des = np.loadtxt(p)
            try:
                dist = distance_f(vec_descriptor, des, distance_choice)
            except:
                continue
            name = os.path.splitext(f)[0]
            ldistances.append((name.split('_')[4],
                               os.path.join(settings.MEDIA_URL, settings.MIR_DATABASE, *p.split(os.sep)[4:6],
                                            name + '.jpg'), dist))
    ldistances.sort(key=operator.itemgetter(2))
    lvoisins = []
    for i in range(top):
        lvoisins.append(ldistances[i])
    return lvoisins






def distance_f(l1, l2, distanceName):
    if distanceName == DescriptorRequests.DistanceChoices.EUCLIDEAN:
        distance = euclidean(l1, l2)
    elif distanceName in [DescriptorRequests.DistanceChoices.CORRELATION,
                          DescriptorRequests.DistanceChoices.CHI_SQUARE,
                          DescriptorRequests.DistanceChoices.INTERSECTION,
                          DescriptorRequests.DistanceChoices.BHATTACHARYA]:

        if distanceName == DescriptorRequests.DistanceChoices.CORRELATION:
            methode = cv2.HISTCMP_CORREL
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName == DescriptorRequests.DistanceChoices.CHI_SQUARE:
            distance = chiSquareDistance(l1, l2)
        elif distanceName == DescriptorRequests.DistanceChoices.INTERSECTION:
            methode = cv2.HISTCMP_INTERSECT
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName == DescriptorRequests.DistanceChoices.BHATTACHARYA:
            distance = bhatta(l1, l2)
    elif distanceName == DescriptorRequests.DistanceChoices.BRUTE_FORCE:
        distance = bruteForceMatching(l1, l2)
    elif distanceName == DescriptorRequests.DistanceChoices.FLANN:
        distance = flann(l1, l2)
    return distance




def Compute_RP(top,class_image_requete, noms_images_proches, descripteur, distance, r,rp_process='rp'):
    rappel_precision=[]
    rp = []
    # position1=int(class_image_requete)//100
    position1=class_image_requete
    for j in range(top):
        # position2=int(noms_images_proches[j])//100
        position2=noms_images_proches[j]
        if position1==position2:
            rappel_precision.append("pertinant")
        else:
            rappel_precision.append("non pertinant")
    for i in range(top):
        j=i
        val=0
        while j>=0:
            if rappel_precision[j]=="pertinant":
                val+=1
            j-=1
        t = get_top(class_image_requete)
        rappel = val/t
        precision = val/(i+1)
        rp.append((rappel,precision))
    mean_r =round(sum(elt[0] for elt in rp)/len(rp)*100,2)
    mean_p = round(sum(elt[1] for elt in rp)/len(rp)*100,2)
    return Display_RP(rp, descripteur, distance, rp_process=rp_process ), mean_r, mean_p, r_precision(rappel_precision, r), f_mesure(mean_p, mean_r)

def f_mesure(p,r):
    return round((2*p*r)/(p+r)  ,2)
def r_precision(rappel_precision, r):
    if not r<= len(rappel_precision):
        r=len(rappel_precision)
    val=0
    for i in range(r):
        if rappel_precision[i]=="pertinant":
            val+=1
    return round(val/r*100,2)
def Display_RP(rp, descripteur, distance, rp_process = 'rp'):
    r, p = zip(*rp)
    if rp_process == 'rp':
        rappel = r
        precision = p
    if rp_process == 'Mrp':
        rappel = ((r + np.roll(r,1))/2.0)[1::2]
        precision = ((p + np.roll(p,1))/2.0)[1::2]
    fig = plt.figure()
    plt.plot(rappel, precision,'C1', label=descripteur)
    plt.xlabel('Rappel')
    plt.ylabel('Précison')
    plt.title(f"{rp_process} descriptor: {descripteur} & distance: {distance}\n"
              f"Total precision : {round(precision[-1]*100,2)}%\n"
              f"Total recall: {round(rappel[-1]*100,2)}%")
    plt.legend()
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    return data


def getkVoisins2_files222222(vec_descriptor, top, descriptor_folder1, descriptor_folder2, distance_choice):
    ldistances = []
    for path, subdir, files in os.walk(os.path.join(settings.MEDIA_ROOT, descriptor_folder1)):
        for f in files:
            p1 = os.path.join(path, f)
            des1 = np.loadtxt(p1)
            p2 = os.path.join(settings.MEDIA_ROOT,descriptor_folder2,*p1.split(os.sep)[4:])
            des2 = np.loadtxt(p2)
            if descriptor_folder1 == 'BGR' or descriptor_folder1 == 'HSV':
                des1 = des1.ravel()
            if descriptor_folder2 == 'BGR' or descriptor_folder2 == 'HSV':
                des2 = des2.ravel()
            featureTOT = np.concatenate([des1,des2])

            dist = distance_f(vec_descriptor, featureTOT, distance_choice)
            name = os.path.splitext(f)[0]
            ldistances.append((name.split('_')[4],
                               os.path.join(settings.MEDIA_URL, settings.MIR_DATABASE, *p1.split(os.sep)[4:6],
                                            name + '.jpg'), dist))
    ldistances.sort(key=operator.itemgetter(2))
    lvoisins = []
    for i in range(top):
        lvoisins.append(ldistances[i])
    return lvoisins

def extractReqFeatures(fileName, algo_choice):
    img = cv2.imread(f'/app{fileName}')
    if algo_choice == DescriptorRequests.DescriptorChoices.BGR:
        return BGR(img), 'BGR'
    if algo_choice == DescriptorRequests.DescriptorChoices.HSV:
        return HSV(img), 'HSV'
    elif algo_choice == DescriptorRequests.DescriptorChoices.SIFT:  # ORB
        return SIFT(img), "SIFT"
    elif algo_choice == DescriptorRequests.DescriptorChoices.ORB:  # ORB
        return ORB(img), "ORB"
    elif algo_choice == DescriptorRequests.DescriptorChoices.GLCM:  # glcm
        return GLCM(img), "GLCM"
    elif algo_choice == DescriptorRequests.DescriptorChoices.LBP:  # lbp
        return LBP(img), "LBP"
    elif algo_choice == DescriptorRequests.DescriptorChoices.HOG:  # hog
        return HOG(img), "HOG"
    elif algo_choice == DescriptorRequests.DescriptorChoices.VGG16:
        return VGG16(fileName), "VGG16"
    elif algo_choice == DescriptorRequests.DescriptorChoices.VGG16_1:
        return VGG16_1(fileName), "VGG16_1"
    elif algo_choice == DescriptorRequests.DescriptorChoices.RESNET101:
        return RESNET101(fileName), "ResNet101"
    elif algo_choice == DescriptorRequests.DescriptorChoices.RESNET101_1:
        return RESNET101_1(fileName), "ResNet101_1"
    elif algo_choice == DescriptorRequests.DescriptorChoices.RESNET50:
        return RESNET50(fileName), "ResNet50"
    elif algo_choice == DescriptorRequests.DescriptorChoices.RESNET50_1:
        return RESNET50_1(fileName), "ResNet50_1"
    elif algo_choice == DescriptorRequests.DescriptorChoices.MOBILENET:
        return MOBILENET(fileName), "MobileNet"
    elif algo_choice == DescriptorRequests.DescriptorChoices.XCEPTION:
        return MOBILENET(fileName), "Xception"

def get_top(class_choice):
    if class_choice == ImageRequests.ClassChoices.ARAIGNEE:
        return TopClassChoices.TOP_SPIDERS
    if class_choice == ImageRequests.ClassChoices.CHIEN:
        return TopClassChoices.TOP_DOGS
    if class_choice == ImageRequests.ClassChoices.OISEAU:
        return TopClassChoices.TOP_BIRDS
    if class_choice == ImageRequests.ClassChoices.POISSON:
        return TopClassChoices.TOP_FISH
    if class_choice == ImageRequests.ClassChoices.SINGE:
        return TopClassChoices.TOP_MONKEY


