from django.conf import settings

def version_processor(request):
    return {
        'VERSION': settings.APP_VERSION
    }
