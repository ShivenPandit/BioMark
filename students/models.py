from django.db import models

class Student(models.Model):
    student_id = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=100)
    department = models.CharField(max_length=100)
    course = models.CharField(max_length=100)
    year = models.CharField(max_length=20)
    semester = models.CharField(max_length=20)
    division = models.CharField(max_length=20)
    roll = models.CharField(max_length=20)
    gender = models.CharField(max_length=10)
    dob = models.DateField(null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    phone = models.CharField(max_length=20, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    teacher = models.CharField(max_length=100, null=True, blank=True)
    photo_sample = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.student_id} - {self.name}"
        
    @property
    def full_name(self):
        return self.name 