from django import forms
from .models import ImageRequests, DescriptorRequests


class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = ImageRequests
        fields = ('title', 'image', 'classification')

class SearchForm(forms.ModelForm):
    class Meta:
        model = DescriptorRequests
        fields = ('descriptor1', 'descriptor2', 'distance', 'top')
