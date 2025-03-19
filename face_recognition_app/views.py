from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from students.models import Student
from attendance.models import Attendance
from .models import TrainingLog
import cv2
import numpy as np
import os
from datetime import datetime
from django.conf import settings
from django.http import JsonResponse
import json

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    
    return render(request, 'face_recognition_app/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def dashboard(request):
    student_count = Student.objects.count()
    attendance_today = Attendance.objects.filter(date=datetime.now().date()).count()
    last_training = TrainingLog.objects.order_by('-trained_at').first()
    
    context = {
        'student_count': student_count,
        'attendance_today': attendance_today,
        'last_training': last_training,
    }
    
    return render(request, 'face_recognition_app/dashboard.html', context)

@login_required
def train_model(request):
    try:
        # Get statistics for the template
        total_students = Student.objects.count()
        students_with_photos = Student.objects.filter(photo_sample=True).count()
        training_count = TrainingLog.objects.count()
        last_training = TrainingLog.objects.order_by('-trained_at').first()
        
        # Count total photos
        total_photos = 0
        for student in Student.objects.filter(photo_sample=True):
            student_photo_dir = os.path.join(settings.MEDIA_ROOT, 'student_photos', str(student.student_id))
            if os.path.exists(student_photo_dir):
                total_photos += len([f for f in os.listdir(student_photo_dir) 
                                  if os.path.isfile(os.path.join(student_photo_dir, f)) 
                                  and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if request.method == 'POST':
            try:
                # Create directories if they don't exist
                model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_model')
                os.makedirs(model_dir, exist_ok=True)
                
                # Get all students with photos
                students = Student.objects.filter(photo_sample=True)
                
                if not students:
                    messages.warning(request, "No students with photos found. Please add student photos first.")
                    return redirect('train_model')
                
                # Check if MEDIA_ROOT exists
                if not os.path.exists(settings.MEDIA_ROOT):
                    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
                    messages.warning(request, f"Media directory was missing and has been created at {settings.MEDIA_ROOT}")
                
                # Check if student_photos directory exists
                student_photos_dir = os.path.join(settings.MEDIA_ROOT, 'student_photos')
                if not os.path.exists(student_photos_dir):
                    os.makedirs(student_photos_dir, exist_ok=True)
                    messages.warning(request, f"Student photos directory was missing and has been created at {student_photos_dir}")
                    return redirect('train_model')
                
                # Verify that student directories exist and contain images
                valid_student_dirs = []
                missing_directories = []
                empty_directories = []
                
                for student in students:
                    student_photo_dir = os.path.join(settings.MEDIA_ROOT, 'student_photos', str(student.student_id))
                    
                    # Create student directory if it doesn't exist
                    if not os.path.exists(student_photo_dir):
                        os.makedirs(student_photo_dir, exist_ok=True)
                        print(f"Created missing directory for student {student.student_id}: {student_photo_dir}")
                        missing_directories.append(student.student_id)
                        continue
                    
                    # Check if directory contains images
                    image_files = [f for f in os.listdir(student_photo_dir) 
                                 if os.path.isfile(os.path.join(student_photo_dir, f)) 
                                 and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if image_files:
                        valid_student_dirs.append((student, student_photo_dir, image_files))
                        print(f"Found {len(image_files)} images for student {student.student_id}")
                    else:
                        empty_directories.append(student.student_id)
                        print(f"No images found for student {student.student_id} in {student_photo_dir}")
                
                if not valid_student_dirs:
                    error_message = "No student photos found. Please add photos for students before training."
                    if missing_directories or empty_directories:
                        error_message += f" Missing directories: {missing_directories}. Empty directories: {empty_directories}."
                    messages.error(request, error_message)
                    return redirect('train_model')
                
                print(f"Found {len(valid_student_dirs)} students with valid photos")
                
                # Dictionary to store face encodings
                known_face_encodings = []
                known_face_names = []
                
                # Debug counters
                total_images = 0
                images_with_faces = 0
                
                # Process each student's photos
                for student, student_photo_dir, image_files in valid_student_dirs:
                    print(f"Processing {len(image_files)} images for student {student.student_id}")
                    
                    for image_file in image_files:
                        try:
                            total_images += 1
                            # Read image
                            img_path = os.path.join(student_photo_dir, image_file)
                            print(f"Processing image: {img_path}")
                            
                            # Simple approach: use OpenCV for face detection
                            try:
                                # Read the image with OpenCV
                                img = cv2.imread(img_path)
                                
                                if img is None:
                                    print(f"Error reading image: {img_path}")
                                    continue
                                
                                # Convert to grayscale for face detection
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                
                                # Use Haar Cascade for face detection
                                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                                if face_cascade.empty():
                                    print("Error: Haar cascade file not found or invalid")
                                    continue
                                    
                                faces = face_cascade.detectMultiScale(
                                    gray,
                                    scaleFactor=1.1,
                                    minNeighbors=2,  # More lenient parameter for training
                                    minSize=(15, 15)  # Smaller minimum face size
                                )
                                
                                if len(faces) > 0:
                                    images_with_faces += 1
                                    print(f"Found {len(faces)} faces with OpenCV in {img_path}")
                                    
                                    # Get the largest face
                                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                                    x, y, w, h = largest_face
                                    
                                    # Create a simple feature vector from the face region
                                    face_roi = gray[y:y+h, x:x+w]
                                    face_roi_resized = cv2.resize(face_roi, (50, 50))
                                    
                                    # Flatten the image to create a simple feature vector
                                    face_encoding = face_roi_resized.flatten()
                                    
                                    # Normalize the encoding
                                    face_encoding = face_encoding / 255.0
                                    
                                    # Store the encoding and student ID
                                    known_face_encodings.append(face_encoding)
                                    known_face_names.append(student.student_id)
                                    print(f"Successfully encoded face for student {student.student_id}")
                                else:
                                    print(f"No faces detected in {img_path}")
                            
                            except Exception as e:
                                print(f"Error processing image with OpenCV: {str(e)}")
                                continue
                                
                        except Exception as e:
                            print(f"Error processing image {image_file}: {str(e)}")
                            continue
                
                print(f"Summary: Processed {total_images} images, found faces in {images_with_faces} images")
                print(f"Generated {len(known_face_encodings)} face encodings for {len(set(known_face_names))} students")
                
                if len(known_face_encodings) > 0:
                    # Save the encodings
                    data = {
                        "encodings": known_face_encodings,
                        "names": known_face_names
                    }
                    
                    # Save with pickle
                    import pickle
                    with open(os.path.join(model_dir, 'face_encodings.pickle'), 'wb') as f:
                        pickle.dump(data, f)
                    
                    # Save the student ID mapping
                    student_id_map = {str(student_id): str(student_id) for student_id in known_face_names}
                    with open(os.path.join(model_dir, 'student_id_map.json'), 'w') as f:
                        json.dump(student_id_map, f)
                    
                    # Log the training
                    TrainingLog.objects.create(
                        trained_by=request.user,
                        status='Success',
                        message=f'Model trained successfully with {len(known_face_encodings)} face images from {len(set(known_face_names))} students'
                    )
                    
                    messages.success(request, f"Face recognition model trained successfully with {len(known_face_encodings)} images!")
                else:
                    raise Exception(f"No faces could be detected in any of the student photos. Processed {total_images} images but found 0 faces.")
                    
            except Exception as e:
                import traceback
                print(f"Training error: {str(e)}")  # Print error to console for debugging
                print(traceback.format_exc())
                TrainingLog.objects.create(
                    trained_by=request.user,
                    status='Failed',
                    message=str(e)
                )
                messages.error(request, f"Error training model: {str(e)}")
            
            return redirect('train_model')
        
        context = {
            'total_students': total_students,
            'students_with_photos': students_with_photos,
            'total_photos': total_photos,
            'training_count': training_count,
            'last_training': last_training,
            'is_training': False
        }
        
        return render(request, 'face_recognition_app/train_model.html', context)
        
    except Exception as e:
        import traceback
        print(f"View error: {str(e)}")  # Print error to console for debugging
        print(traceback.format_exc())
        messages.error(request, "An unexpected error occurred. Please try again or contact the administrator.")
        return redirect('dashboard')

@login_required
def face_recognition_view(request):
    # Check if model exists
    model_path = os.path.join(settings.MEDIA_ROOT, 'trained_model', 'face_encodings.pickle')
    model_exists = os.path.exists(model_path)
    
    # Get recent attendance records
    try:
        recent_attendance = Attendance.objects.filter(
            date=datetime.now().date()
        ).order_by('-time')[:5]
    except Exception as e:
        print(f"Error fetching recent attendance: {str(e)}")
        recent_attendance = []
    
    context = {
        'model_exists': model_exists,
        'recent_attendance': recent_attendance
    }
    
    return render(request, 'face_recognition_app/face_recognition.html', context)

@login_required
def process_face_recognition(request):
    if request.method == 'POST':
        try:
            # Check if image data is provided
            if 'image' not in request.POST and 'image_data' not in request.POST:
                return JsonResponse({'status': 'error', 'message': 'No image data provided'})
            
            # Get image data from POST request (support both parameter names)
            image_data = request.POST.get('image', request.POST.get('image_data', ''))
            
            # Remove the data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            import base64
            from PIL import Image
            import io
            
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert PIL image to OpenCV format
                import numpy as np
                img = np.array(image)
                img = img[:, :, ::-1].copy()  # RGB to BGR for OpenCV
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': f'Error processing image: {str(e)}'})
            
            # Check if model exists
            model_path = os.path.join(settings.MEDIA_ROOT, 'trained_model', 'face_encodings.pickle')
            if not os.path.exists(model_path):
                return JsonResponse({'status': 'error', 'message': 'Face recognition model not found. Please train the model first.'})
            
            # Load the face recognition model
            import pickle
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    known_face_encodings = data["encodings"]
                    known_face_names = data["names"]
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': f'Error loading model: {str(e)}'})
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )
            
            if len(faces) == 0:
                return JsonResponse({'status': 'error', 'message': 'No face detected in the image'})
            
            # Process each detected face
            recognized_students = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Resize to match the training size
                face_roi_resized = cv2.resize(face_roi, (50, 50))
                
                # Flatten and normalize
                face_encoding = face_roi_resized.flatten() / 255.0
                
                # Compare with known faces
                matches = []
                face_distances = []
                
                for i, known_encoding in enumerate(known_face_encodings):
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(face_encoding - known_encoding)
                    face_distances.append(distance)
                    
                    # Consider it a match if distance is below threshold
                    if distance < 15.0:  # Adjusted threshold to be more lenient
                        matches.append(i)
                
                if matches:
                    # Find the closest match
                    best_match_idx = np.argmin(face_distances)
                    student_id = known_face_names[best_match_idx]
                    
                    # Calculate confidence score (inverse of distance, normalized)
                    confidence = max(0, 100 - (face_distances[best_match_idx] * 10))
                    confidence = min(confidence, 100)  # Cap at 100%
                    
                    # Log recognition details
                    print(f"Recognition success - Student ID: {student_id}, Distance: {face_distances[best_match_idx]:.2f}, Confidence: {confidence:.2f}%")
                    
                    # Get student details
                    try:
                        student = Student.objects.get(student_id=student_id)
                        
                        # Mark attendance
                        attendance, created = Attendance.objects.get_or_create(
                            student=student,
                            date=datetime.now().date(),
                            defaults={'time': datetime.now().time()}
                        )
                        
                        if not created:
                            # Update time for existing attendance
                            attendance.time = datetime.now().time()
                            attendance.save()
                        
                        # Add to recognized students
                        student_name = getattr(student, 'full_name', f"Student {student_id}")
                        student_course = getattr(student, 'course', 'N/A')
                        
                        recognized_students.append({
                            'student_id': student_id,
                            'name': student_name,
                            'course': student_course,
                            'confidence': round(confidence, 2),
                            'attendance_marked': True
                        })
                    except Student.DoesNotExist:
                        recognized_students.append({
                            'student_id': student_id,
                            'name': f"Unknown (ID: {student_id})",
                            'course': 'N/A',
                            'confidence': round(confidence, 2),
                            'attendance_marked': False,
                            'error': 'Student not found in database'
                        })
            
            if recognized_students:
                return JsonResponse({
                    'status': 'success',
                    'message': f"Recognized {len(recognized_students)} student(s)",
                    'students': recognized_students
                })
            else:
                # No students were recognized from the detected faces
                # Get minimum distance info for debugging if available
                if 'face_distances' in locals() and len(face_distances) > 0:
                    min_distance_idx = np.argmin(face_distances)
                    min_distance = face_distances[min_distance_idx]
                    closest_student_id = known_face_names[min_distance_idx]
                    print(f"Face detected but not recognized. Closest match: Student ID {closest_student_id}, Distance: {min_distance:.2f} (threshold: 15.0)")
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Face detected but not recognized. Closest match had distance {min_distance:.2f} (threshold: 15.0)'
                    })
                else:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'No students recognized'
                    })
                
        except Exception as e:
            import traceback
            print(f"Recognition error: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required
def developer_view(request):
    return render(request, 'face_recognition_app/developer.html')

@login_required
def help_support(request):
    return render(request, 'face_recognition_app/help_support.html') 