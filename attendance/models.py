from django.db import models
from students.models import Student

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    time = models.TimeField()
    status = models.CharField(max_length=10, default='Present')
    
    class Meta:
        unique_together = ['student', 'date']
        
    def __str__(self):
        return f"{self.student.name} - {self.date} - {self.status}" 