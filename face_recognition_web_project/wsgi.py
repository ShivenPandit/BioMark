"""
WSGI config for face_recognition_web_project project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_recognition_web_project.settings')

application = get_wsgi_application() 