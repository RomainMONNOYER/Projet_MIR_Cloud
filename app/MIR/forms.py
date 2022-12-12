from django import forms
from .models import ImageRequests


class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = ImageRequests
        fields = ('title', 'image')

class SearchForm(forms.Form):
    sift = forms.BooleanField()
    orb = forms.BooleanField()
    HSV = forms.BooleanField()
    class Meta:
        model = ImageRequests
        fields = ('sift', 'orb', 'HSV')