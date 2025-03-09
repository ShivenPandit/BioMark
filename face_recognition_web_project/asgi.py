"""
ASGI config for face_recognition_web_project project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_recognition_web_project.settings')

application = get_asgi_application() 