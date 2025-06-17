from django.contrib import admin
from . models import VideoCount, Accuracy, TrustScore

# Register your models here.
admin.site.register(VideoCount)
admin.site.register(Accuracy)
admin.site.register(TrustScore)