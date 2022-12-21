import enum

from django.db import models


class ImageRequests(models.Model):
    class ClassChoices(models.IntegerChoices):
        ARAIGNEE = 0
        CHIEN = 1
        OISEAU = 2
        POISSON = 3
        SINGE = 4
    class SubClassChoices(models.IntegerChoices):
    #     # BARN_SPIDER = 0
    #     # WOLF_SPIDER = 1
    #     # TRAP_DOOR_SPIDER = 2
    #     # ORB_WEAVING_SPIDER = 3
    #     # GARDEN_SPIDER = 4
    #     # TARANTULA = 5
        SIBERIAN_HUSKY= 0
        LABRADOR_RETRIEVER = 1
        BOXER = 2
        CHIHUAHUA = 3
        GOLDER_RETRIEVER = 4
        ROTTWEILER = 5

    title = models.CharField(max_length=255)
    classification = models.IntegerField(default=ClassChoices.ARAIGNEE, choices=ClassChoices.choices)
    subclassification = models.IntegerField(default=SubClassChoices.SIBERIAN_HUSKY, choices=SubClassChoices.choices)
    date_upload = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='images/requests')

    def __str__(self):
        return self.title


class DescriptorRequests(models.Model):
    class DistanceChoices(models.TextChoices):
        EUCLIDEAN = 'Euclidean'
        CHI_SQUARE = 'Chi Square'
        BHATTACHARYA = 'Bhattacharya'
        FLANN = 'Flann'
        BRUTE_FORCE = 'Brute Force'
        INTERSECTION = 'Intersection'
        CORRELATION = 'Correlation'
    class DescriptorChoices(models.TextChoices):
        HSV = 'HSV'
        BGR = 'BGR'
        SIFT = 'SIFT'
        ORB = 'ORB'
        GLCM = 'GLCM'
        LBP = 'LBP'
        HOG = 'HOG'
        VGG16 = 'VGG16'
        VGG16_1 = 'VGG16_1'


    BGR = models.BooleanField(default=False)
    HSV = models.BooleanField(default=False)
    SIFT = models.BooleanField(default=False)
    ORB = models.BooleanField(default=False)
    GLCM = models.BooleanField(default=False)
    LBP = models.BooleanField(default=False)
    HOG = models.BooleanField(default=False)
    VGG16 = models.BooleanField(default=False)
    VGG16_1 = models.BooleanField(default=False)
    distance = models.CharField(default=DistanceChoices.EUCLIDEAN, choices=DistanceChoices.choices, max_length=15)
    descriptor1 = models.CharField(default=DescriptorChoices.BGR, choices=DescriptorChoices.choices, max_length=15)
    descriptor2 = models.CharField(default=DescriptorChoices.BGR, choices=DescriptorChoices.choices, max_length=15)
    top = models.IntegerField(default=5)

    def __str__(self):
        return str({"BGR": self.BGR,
                    "HSV": self.HSV,
                    "SIFT": self.SIFT,
                    "ORB": self.ORB,
                    "GLCM": self.GLCM,
                    "LBP": self.LBP,
                    "HOG": self.HOG,
                    "VGG16": self.VGG16,
                    "Distance": self.distance,
                    "Top": self.top,
                    })
