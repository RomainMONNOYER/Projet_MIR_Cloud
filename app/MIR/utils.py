# Defintion de toute les fonctions à appeller dans l'interface
import json
import operator
import os
import cv2
import numpy as np
from django.conf import settings
from skimage.transform import resize
from skimage import io, color, img_as_ubyte
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern

import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from .distances import euclidean, chiSquareDistance, bhatta, bruteForceMatching, flann
from .models import DescriptorRequests


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

def VGG16_1(fileName):
    image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature =  Model(inputs=settings.VGG16.input, outputs=settings.VGG16.layers[-2].output).predict(image)  # predict the probability
    return np.array(feature[0])


def extractReqFeatures(fileName, algo_choice):
    img = cv2.imread(f'/app{fileName}')
    resized_img = resize(img, (128 * 4, 64 * 4))
    print(algo_choice)
    if algo_choice.BGR:
        return BGR(img), 'BGR'
    elif algo_choice.HSV:
        return HSV(img), "HSV"
    elif algo_choice.SIFT:  # SIFT
        sift = cv2.SIFT_create()
        kps, vect_features = sift.detectAndCompute(img, None)
    elif algo_choice.ORB:  # ORB
        return ORB(img), "ORB"
    elif algo_choice.GLCM:  # glcm
        return GLCM(img), "GLCM"
    elif algo_choice.LBP:  # lbp
        return LBP(img), "LBP"
    elif algo_choice.HOG:  # hog
        return HOG(img), "HOG"
    elif algo_choice.VGG16:
        return VGG16(fileName), "VGG16"
    elif algo_choice.VGG16_1:
        return VGG16_1(fileName), "VGG16_1"


def getkVoisins2_files(vec_descriptor, top, descriptor_folder, distance_choice):
    ldistances = []
    for path, subdir, files in os.walk(os.path.join(settings.MEDIA_ROOT, descriptor_folder)):
        for f in files:
            p = os.path.join(path, f)
            des = np.loadtxt(p)
            dist = distance_f(vec_descriptor, des, distance_choice)
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
