Deploying to Render

Prerequisites
- Render account
- Repo pushed to GitHub/GitLab/Bitbucket

What’s included
- Dockerfile (Python 3.11 slim, system libs for OpenCV/dlib)
- start.sh (migrate, collectstatic, start Gunicorn)
- render.yaml (web service + free Postgres database)
- Production tweaks in face_recognition_web_project/settings.py
- gunicorn, whitenoise added to requirements.txt

Steps
1) Push this project to a Git host.
2) In Render: New → Blueprint → select this repo.
3) Create resources; first build will start automatically.
4) After deploy, visit the Render URL.

Notes
- A free Postgres database is provisioned; Django picks it up via DATABASE_URL.
- If DATABASE_URL is absent, it falls back to SQLite (local dev).
- Static files are served by WhiteNoise after collectstatic.
- For admin, run a one-off shell and: python manage.py createsuperuser

