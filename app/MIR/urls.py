from django.urls import path

from . import views

urlpatterns = [
    path('upload_1desc/', views.image_upload_view, name='upload_image'),
    path('upload_2desc/', views.image_upload_view2, name='upload_image2'),
    path('db/', views.image_from_db, name='db_images'),
    path('history/', views.image_history, name='db_history'),
    path('search/<int:pk>', views.image_search),
    path('search-2descriptors/<int:pk>', views.image_search2, name='search-2descriptor'),
    path('home', views.home, name='manuel'),
]
