from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.image_upload_view),
    path('search/<int:pk>', views.image_search),
]
