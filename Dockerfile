FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System deps needed by opencv-python and dlib wheels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
RUN pip install --upgrade pip setuptools wheel cmake \
    && pip install --no-cache-dir --prefer-binary dlib==19.24.2 \
    && pip install -r requirements.txt

COPY . .

# Normalize potential CRLF in scripts when repo is authored on Windows
RUN sed -i 's/\r$//' start.sh

# Expose port Render provides via $PORT
ENV DJANGO_SETTINGS_MODULE=face_recognition_web_project.settings

# Make start script executable
RUN chmod +x start.sh

CMD ["./start.sh"]


