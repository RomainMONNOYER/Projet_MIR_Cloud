#Defintion de toute les fonctions à appeller dans l'interface
import json
import operator
import os
import cv2
import numpy as np
from django.conf import settings
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern

from tensorflow import keras
from keras.applications import vgg16
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input

# def generateHistogramme_HSV(filenames, progressBar):
#     if not os.path.isdir("HSV"):
#         os.mkdir("HSV")
#     i=0
#     for path in os.listdir(filenames):
#         img = cv2.imread(filenames+"/"+path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         histH = cv2.calcHist([img],[0],None,[256],[0,256])
#         histS = cv2.calcHist([img],[1],None,[256],[0,256])
#         histV = cv2.calcHist([img],[2],None,[256],[0,256])
#         feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)
#
#         num_image, _ = path.split(".")
#         np.savetxt("HSV/"+str(num_image)+".txt" ,feature)
#
#         progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
#         i+=1
#     print("indexation Hist HSV terminée !!!!")

# def generateHistogramme_Color(filenames, progressBar):
#     if not os.path.isdir("BGR"):
#         os.mkdir("BGR")
#     i=0
#     for path in os.listdir(filenames):
#         img = cv2.imread(filenames+"/"+path)
#         histB = cv2.calcHist([img],[0],None,[256],[0,256])
#         histG = cv2.calcHist([img],[1],None,[256],[0,256])
#         histR = cv2.calcHist([img],[2],None,[256],[0,256])
#         feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
#
#         num_image, _ = path.split(".")
#         np.savetxt("BGR/"+str(num_image)+".txt" ,feature)
#         progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
#         i+=1
#     print("indexation Hist Couleur terminée !!!!")
#
# def generateSIFT(filenames, progressBar):
#     if not os.path.isdir("SIFT"):
#         os.mkdir("SIFT")
#     i=0
#     for path in os.listdir(filenames):
#         img = cv2.imread(filenames+"/"+path)
#         featureSum = 0
#         sift = cv2.SIFT_create()
#         kps , des = sift.detectAndCompute(img,None)
#
#         num_image, _ = path.split(".")
#         np.savetxt("SIFT/"+str(num_image)+".txt" ,des)
#         progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
#
#         featureSum += len(kps)
#         i+=1
#     print("Indexation SIFT terminée !!!!")
#
#
# def generateORB(filenames, progressBar):
#     if not os.path.isdir("ORB"):
#         os.mkdir("ORB")
#     i=0
#     for path in os.listdir(filenames):
#         img = cv2.imread(filenames+"/"+path)
#         orb = cv2.ORB_create()
#         key_point1,descrip1 = orb.detectAndCompute(img,None)
#
#         num_image, _ = path.split(".")
#         np.savetxt("ORB/"+str(num_image)+".txt" ,descrip1 )
#         progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
#         i+=1
#     print("indexation ORB terminée !!!!")

def extractReqFeatures(fileName,algo_choice):
    img = cv2.imread(f'/app{fileName}')
    resized_img = resize(img, (128*4, 64*4))
    if algo_choice.BGR: #Histo Couleurs
        print(algo_choice)
        histB = cv2.calcHist([img],[0],None,[256],[0,256])
        histG = cv2.calcHist([img],[1],None,[256],[0,256])
        histR = cv2.calcHist([img],[2],None,[256],[0,256])
        vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)

        BGR_path = os.path.join(settings.MEDIA_ROOT, 'BGR', 'data.txt')
        print(BGR_path)
        with open(BGR_path, 'r') as r:
            json_data = json.load(r)
        return vect_features, json_data

    elif algo_choice.HSV: # Histo HSV
        print(algo_choice)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([hsv],[0],None,[256],[0,256])
        histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
        histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
        vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        HSV_path = os.path.join(settings.MEDIA_ROOT, 'HSV', 'data.txt')
        print(HSV_path)
        with open(HSV_path, 'r') as r:
            json_data = json.load(r)
        return vect_features, json_data

    elif algo_choice.SIFT: #SIFT
        print(algo_choice)
        sift = cv2.SIFT_create()
        kps , vect_features = sift.detectAndCompute(img,None)

    elif algo_choice.ORB: #ORB
        print(algo_choice)
        orb = cv2.ORB_create()
        key_point1,vect_features = orb.detectAndCompute(img,None)
    elif algo_choice.GLCM: #glcm
        distances=[1,-1]
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)
        glcmMatrix = greycomatrix(gray, distances=distances, angles=angles,
                                  normed=True)
        glcmProperties1 = greycoprops(glcmMatrix,'contrast').ravel()
        glcmProperties2 = greycoprops(glcmMatrix,'dissimilarity').ravel()
        glcmProperties3 = greycoprops(glcmMatrix,'homogeneity').ravel()
        glcmProperties4 = greycoprops(glcmMatrix,'energy').ravel()
        glcmProperties5 = greycoprops(glcmMatrix,'correlation').ravel()
        glcmProperties6 = greycoprops(glcmMatrix,'ASM').ravel()
        vect_features = np.array([glcmProperties1,glcmProperties2,glcmProperties3,glcmProperties4,glcmProperties5,glcmProperties6]).ravel()
    elif algo_choice.LBP: #lbp
        points=8
        radius=1
        method='default'
        subSize=(70,70)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(350,350))
        fullLBPmatrix = local_binary_pattern(img,points,radius,method)
        histograms = []
        for k in range(int(fullLBPmatrix.shape[0]/subSize[0])):
            for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
                subVector =fullLBPmatrix[k*subSize[0]:(k+1)*subSize[0],j*subSize[1]:(j+1)*subSize[1]].ravel()
                subHist,edges =np.histogram(subVector,bins=int(2**points),range=(0,2**points))
                histograms = np.concatenate((histograms,subHist),axis=None)
        LBP_path = os.path.join(settings.MEDIA_ROOT, 'LBP', 'data.txt')
        with open(LBP_path, 'r') as r:
            json_data = json.load(r)
        return histograms, json_data

    elif algo_choice.HOG: #hog
        cellSize = (25,25)
        blockSize = (50,50)
        blockStride = (25,25)
        nBins = 9
        winSize = (350,350)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,winSize)
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBins)
        vect_features = hog.compute(image)
    elif algo_choice.VGG16:
        model=settings.VGG16
        print(fileName)
        image = tf.keras.utils.load_img(f'/app{fileName}', target_size=(224, 224))
        image = tf.keras.utils.img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image) # predict the probability
        vect_features = np.array(feature[0])

        VGG16_path = os.path.join(settings.MEDIA_ROOT, 'VGG16', 'data.txt')
        print(VGG16_path)
        with open(VGG16_path, 'r') as r:
            json_data = json.load(r)
        return vect_features, json_data

        # np.savetxt(f"Methode_{str(algo_choice)}_requete.txt", vect_features)
    # # print(vect_features)
    # return vect_features


def euclidianDistance(l1,l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    distance=np.linalg.norm(l1-l2)
    return distance
def getkVoisins(lfeatures, vec_descriptor, top) :
    ldistances = []
    for k in lfeatures.keys():
        dist = euclidianDistance(vec_descriptor, lfeatures[k][1])
        ldistances.append((k, lfeatures[k], dist))
    ldistances.sort(key=operator.itemgetter(2))
    lvoisins = []
    for i in range(top):
        lvoisins.append(ldistances[i])
    return lvoisins
