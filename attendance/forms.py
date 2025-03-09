from django import forms
from .models import Attendance
from students.models import Student
from datetime import datetime

class AttendanceForm(forms.ModelForm):
    class Meta:
        model = Attendance
        fields = ['student', 'date', 'time', 'status']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
            'time': forms.TimeInput(attrs={'type': 'time'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initial.get('date'):
            self.initial['date'] = datetime.now().date()
        if not self.initial.get('time'):
            self.initial['time'] = datetime.now().time().strftime('%H:%M')

class AttendanceFilterForm(forms.Form):
    date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'type': 'date'})
    )
    student = forms.ModelChoiceField(
        queryset=Student.objects.all(),
        required=False
    ) 