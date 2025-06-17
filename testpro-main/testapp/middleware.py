class CacheControlMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        if request.path.startswith('/static/'):
            response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return response
