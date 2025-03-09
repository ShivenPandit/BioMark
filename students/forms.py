from django import forms
from .models import Student

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = [
            'student_id', 'name', 'department', 'course', 'year', 
            'semester', 'division', 'roll', 'gender', 'dob', 
            'email', 'phone', 'address', 'teacher'
        ]
        widgets = {
            'dob': forms.DateInput(attrs={'type': 'date'}),
            'address': forms.Textarea(attrs={'rows': 3}),
        } 