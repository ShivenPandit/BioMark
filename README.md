# BioMark - Face Recognition Attendance System

BioMark is a web-based facial recognition attendance system built with Django that automates the process of taking and managing student attendance using facial recognition technology.

## Features

- 👤 Facial Recognition Based Attendance
- 📊 Real-time Attendance Tracking
- 👥 Student Management System
- 📱 Web-based Interface
- 📈 Attendance Reports and Analytics
- 🔒 Secure Authentication System

## Technology Stack

- **Backend Framework:** Django 5.1.7
- **Face Recognition:** face-recognition 1.3.0, dlib 19.24.2
- **Image Processing:** OpenCV 4.8.1
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript
- **Additional Libraries:**
  - NumPy 1.26.3
  - Pillow 10.1.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BioMark.git
cd BioMark
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run database migrations:
```bash
python manage.py migrate
```

5. Create a superuser (admin):
```bash
python manage.py createsuperuser
```

6. Start the development server:
```bash
python manage.py runserver
```

7. Access the application at `http://localhost:8000`

## Project Structure

```
BioMark/
├── attendance/           # Attendance management app
├── face_recognition_app/ # Core face recognition functionality
├── face_recognition_web_project/ # Project settings
├── media/               # Media files storage
├── static/              # Static files (Images)
├── students/            # Student management app
├── templates/           # HTML templates
├── manage.py           # Django management script
└── requirements.txt    # Project dependencies
```

## Usage

1. Log in to the admin panel using superuser credentials
2. Add students to the system with their photos
3. Start taking attendance using the facial recognition feature
4. View and manage attendance records
5. Generate attendance reports

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request


## Acknowledgments

- Face Recognition library by Adam Geitgey
- Django Framework
- OpenCV community

## Support

For support, please open an issue in the GitHub repository. 
