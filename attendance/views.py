from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from .models import Attendance
from students.models import Student
from .forms import AttendanceForm, AttendanceFilterForm
import csv
from datetime import datetime
import io

@login_required
def attendance_list(request):
    form = AttendanceFilterForm(request.GET or None)
    attendances = Attendance.objects.all().order_by('-date', '-time')
    
    if form.is_valid():
        if form.cleaned_data.get('date'):
            attendances = attendances.filter(date=form.cleaned_data['date'])
        if form.cleaned_data.get('student'):
            attendances = attendances.filter(student=form.cleaned_data['student'])
    
    return render(request, 'attendance/attendance_list.html', {
        'attendances': attendances,
        'form': form
    })

@login_required
def attendance_create(request):
    if request.method == 'POST':
        form = AttendanceForm(request.POST)
        if form.is_valid():
            attendance = form.save()
            messages.success(request, 'Attendance record created successfully!')
            return redirect('attendance_list')
    else:
        form = AttendanceForm()
    
    return render(request, 'attendance/attendance_form.html', {'form': form})

@login_required
def attendance_update(request, pk):
    attendance = get_object_or_404(Attendance, pk=pk)
    
    if request.method == 'POST':
        form = AttendanceForm(request.POST, instance=attendance)
        if form.is_valid():
            form.save()
            messages.success(request, 'Attendance record updated successfully!')
            return redirect('attendance_list')
    else:
        form = AttendanceForm(instance=attendance)
    
    return render(request, 'attendance/attendance_form.html', {'form': form})

@login_required
def attendance_delete(request, pk):
    attendance = get_object_or_404(Attendance, pk=pk)
    
    if request.method == 'POST':
        attendance.delete()
        messages.success(request, 'Attendance record deleted successfully!')
        return redirect('attendance_list')
    
    return render(request, 'attendance/attendance_confirm_delete.html', {'attendance': attendance})

@login_required
def export_attendance_csv(request):
    form = AttendanceFilterForm(request.GET or None)
    attendances = Attendance.objects.all().order_by('-date', '-time')
    
    if form.is_valid():
        if form.cleaned_data.get('date'):
            attendances = attendances.filter(date=form.cleaned_data['date'])
        if form.cleaned_data.get('student'):
            attendances = attendances.filter(student=form.cleaned_data['student'])
    
    # Create CSV file
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['Student ID', 'Name', 'Date', 'Time', 'Status'])
    
    for attendance in attendances:
        writer.writerow([
            attendance.student.student_id,
            attendance.student.name,
            attendance.date,
            attendance.time,
            attendance.status
        ])
    
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename=attendance_{datetime.now().strftime("%Y%m%d")}.csv'
    
    return response 