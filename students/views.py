from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import Student
from .forms import StudentForm
import cv2
import os
from django.conf import settings
from django.db import models

@login_required
def student_list(request):
    students = Student.objects.all()
    
    # Search functionality
    search_query = request.GET.get('q', '')
    if search_query:
        students = students.filter(
            models.Q(student_id__icontains=search_query) |
            models.Q(name__icontains=search_query) |
            models.Q(department__icontains=search_query) |
            models.Q(course__icontains=search_query)
        )
    
    # Sorting
    sort_by = request.GET.get('sort', 'name')
    if sort_by == 'id':
        students = students.order_by('student_id')
    elif sort_by == 'department':
        students = students.order_by('department', 'name')
    elif sort_by == 'course':
        students = students.order_by('course', 'name')
    else:  # Default sort by name
        students = students.order_by('name')
    
    return render(request, 'students/student_list.html', {
        'students': students,
        'search_query': search_query,
        'current_sort': sort_by
    })

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
        # Delete student photos from filesystem
        student_photo_dir = os.path.join(settings.MEDIA_ROOT, 'student_photos', str(student.student_id))
        if os.path.exists(student_photo_dir):
            try:
                import shutil
                shutil.rmtree(student_photo_dir)
                print(f"Deleted photo directory for student {student.student_id}: {student_photo_dir}")
            except Exception as e:
                print(f"Error deleting student photos: {str(e)}")
        
        # Delete student record (will cascade to attendance records)
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
        try:
            # Get photos data from form
            photos_data = request.POST.get('photos', '')
            
            if not photos_data:
                messages.error(request, 'No photo data received. Please try again.')
                return redirect('take_photo', student_id=student.student_id)
            
            # Parse the JSON array of photo data
            import json
            import base64
            from PIL import Image
            import io
            
            photos = json.loads(photos_data)
            
            # Save each photo
            for i, photo_data in enumerate(photos):
                try:
                    # Remove the data URL prefix
                    if photo_data.startswith('data:image'):
                        photo_data = photo_data.split(',')[1]
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(photo_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Save image
                    photo_filename = f"photo_{i+1}.jpg"
                    photo_path = os.path.join(student_dir, photo_filename)
                    image.save(photo_path, 'JPEG')
                    print(f"Saved photo {i+1} for student {student.student_id}")
                except Exception as e:
                    print(f"Error saving photo {i+1}: {str(e)}")
            
            # Update student record
            student.photo_sample = True
            student.save()
            
            messages.success(request, f"{len(photos)} photos captured successfully!")
            return redirect('student_detail', student_id=student.student_id)
        except Exception as e:
            import traceback
            print(f"Error processing photos: {str(e)}")
            print(traceback.format_exc())
            messages.error(request, f"Error saving photos: {str(e)}")
            return redirect('take_photo', student_id=student.student_id)
    
    return render(request, 'students/take_photo.html', {'student': student}) 