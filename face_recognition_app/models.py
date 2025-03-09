from django.db import models
from django.contrib.auth.models import User

class TrainingLog(models.Model):
    trained_at = models.DateTimeField(auto_now_add=True)
    trained_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    status = models.CharField(max_length=20, default='Success')
    message = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"Training on {self.trained_at.strftime('%Y-%m-%d %H:%M')}" 