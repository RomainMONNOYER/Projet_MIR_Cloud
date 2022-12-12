from django.db import models

class ImageRequests(models.Model):
    title = models.CharField(max_length=255)
    date_upload = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='images/requests')

    def __str__(self):
        return self.title