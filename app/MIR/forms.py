from django import forms
from .models import ImageRequests, DescriptorRequests


class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = ImageRequests
        fields = ('title', 'image', 'classification', 'subclassification')

class SearchForm(forms.ModelForm):
    class Meta:
        model = DescriptorRequests
        fields = ('BGR', 'HSV', 'SIFT', 'ORB','GLCM', 'LBP', 'HOG','VGG16', "VGG16_1", 'distance', 'top')
