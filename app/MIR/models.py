import enum
from functools import partial

from django.db import models


class ImageRequests(models.Model):
    title = models.CharField(max_length=255)
    date_upload = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='images/requests')

    def __str__(self):
        return self.title


from .distances import euclidean, chiSquareDistance, bhatta, flann
class DescriptorRequests(models.Model):
    class DistanceChoices(models.TextChoices):
        EUCLIDIAN = 'Euclidean'
        CHI = 'Chi Square'
        BHATTA = 'Bhatta'
        FLANN = 'Flann'


    HSV = models.BooleanField(default=False)
    BGR = models.BooleanField(default=False)
    SIFT = models.BooleanField(default=False)
    ORB = models.BooleanField(default=False)
    GLCM = models.BooleanField(default=False)
    LBP = models.BooleanField(default=False)
    HOG = models.BooleanField(default=False)
    VGG16 = models.BooleanField(default=False)
    distance = models.CharField(default=DistanceChoices.EUCLIDIAN, choices=DistanceChoices.choices, max_length=15)
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
