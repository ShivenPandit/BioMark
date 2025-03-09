from django.urls import path
from . import views

urlpatterns = [
    path('', views.student_list, name='student_list'),
    path('create/', views.student_create, name='student_create'),
    path('<str:student_id>/', views.student_detail, name='student_detail'),
    path('<str:student_id>/update/', views.student_update, name='student_update'),
    path('<str:student_id>/delete/', views.student_delete, name='student_delete'),
    path('<str:student_id>/take_photo/', views.take_photo, name='take_photo'),
] 