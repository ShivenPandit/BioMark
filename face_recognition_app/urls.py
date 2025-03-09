from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('train-model/', views.train_model, name='train_model'),
    path('face-recognition/', views.face_recognition_view, name='face_recognition'),
    path('process-face-recognition/', views.process_face_recognition, name='process_face_recognition'),
    path('developer/', views.developer_view, name='developer'),
    path('help-support/', views.help_support, name='help_support'),
] 