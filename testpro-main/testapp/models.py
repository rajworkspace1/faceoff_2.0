from django.db import models

# Create your models here.

class VideoCount(models.Model):
    count = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Video uploads: {self.count}"
    
class Accuracy(models.Model):
    module_name = models.CharField(max_length=100)
    video_uuid = models.CharField(max_length=255)  
    accuracy_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.module_name} | {self.video_uuid} | Accuracy: {self.accuracy_score:.2f} | {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"

class TrustScore(models.Model):
    module_name = models.CharField(max_length=100)
    video_uuid = models.CharField(max_length=255)
    trust_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.module_name} | {self.video_uuid} | Trust: {self.trust_score:.2f} | {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"