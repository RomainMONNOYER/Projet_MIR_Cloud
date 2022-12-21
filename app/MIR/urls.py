from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.image_upload_view),
    path('db/', views.image_from_db),
    path('history/', views.image_history),
    path('search/<int:pk>', views.image_search),
    path('search2/<int:pk>', views.image_search2),
]
