from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import Student
from .forms import StudentForm
import cv2
import os
from django.conf import settings

@login_required
def student_list(request):
    students = Student.objects.all()
    return render(request, 'students/student_list.html', {'students': students})

@login_required
def student_detail(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    return render(request, 'students/student_detail.html', {'student': student})

@login_required
def student_create(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():
            student = form.save()
            messages.success(request, 'Student record created successfully!')
            return redirect('student_detail', student_id=student.student_id)
    else:
        form = StudentForm()
    
    return render(request, 'students/student_form.html', {'form': form, 'title': 'Add Student'})

@login_required
def student_update(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    
    if request.method == 'POST':
        form = StudentForm(request.POST, instance=student)
        if form.is_valid():
            form.save()
            messages.success(request, 'Student record updated successfully!')
            return redirect('student_detail', student_id=student.student_id)
    else:
        form = StudentForm(instance=student)
    
    return render(request, 'students/student_form.html', {'form': form, 'title': 'Update Student'})

@login_required
def student_delete(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student record deleted successfully!')
        return redirect('student_list')
    
    return render(request, 'students/student_confirm_delete.html', {'student': student})

@login_required
def take_photo(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    
    # Create directory for student photos if it doesn't exist
    student_dir = os.path.join(settings.MEDIA_ROOT, 'student_photos', str(student.student_id))
    os.makedirs(student_dir, exist_ok=True)
    
    if request.method == 'POST':
        # Logic to capture photos using webcam will be implemented with JavaScript
        # This endpoint will handle the uploaded photos
        student.photo_sample = True
        student.save()
        messages.success(request, 'Photos captured successfully!')
        return redirect('student_detail', student_id=student.student_id)
    
    return render(request, 'students/take_photo.html', {'student': student}) 