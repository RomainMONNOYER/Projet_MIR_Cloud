import enum

from django.db import models


class ImageRequests(models.Model):
    class ClassChoices(models.IntegerChoices):
        ARAIGNEE = 0
        CHIEN = 1
        OISEAU = 2
        POISSON = 3
        SINGE = 4

    title = models.CharField(max_length=255)
    classification = models.IntegerField(default=ClassChoices.ARAIGNEE, choices=ClassChoices.choices)
    is_database_img = models.BooleanField(default=False)
    image = models.ImageField(upload_to='images/requests')
    date_upload = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return {"Title": self.title,
                "Class": self.classification,}


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
        BGR = 'BGR'
        HSV = 'HSV'
        SIFT = 'SIFT'
        ORB = 'ORB'
        GLCM = 'GLCM'
        LBP = 'LBP'
        HOG = 'HOG'
        VGG16 = 'VGG16'
        VGG16_1 = 'VGG16_1'
        RESNET101 = 'RESNET101'
        RESNET101_1 = 'RESNET101_1'
        RESNET50 = 'RESNET50'
        RESNET50_1 = 'RESNET50_1'
        MOBILENET = 'MOBILENET'
        XCEPTION = 'XCEPTION'

    class TopChoices(models.IntegerChoices):
        TOP_5=5
        TOP_10=10
        TOP_20=20
        TOP_50=50
        TOP_100=100
        TOP_200=200
        TOP_MAX=0

    descriptor1 = models.CharField(default=DescriptorChoices.BGR, choices=DescriptorChoices.choices, max_length=15)
    descriptor2 = models.CharField(choices=DescriptorChoices.choices, max_length=15, blank=True)
    distance = models.CharField(default=DistanceChoices.EUCLIDEAN, choices=DistanceChoices.choices, max_length=15)
    top = models.IntegerField(default=TopChoices.TOP_5, choices=TopChoices.choices)

    def __str__(self):
        return str({"Descriptor1": self.descriptor1,
                    "Descriptor2": self.descriptor2,
                    "Distance": self.distance,
                    "Top": self.top,})
