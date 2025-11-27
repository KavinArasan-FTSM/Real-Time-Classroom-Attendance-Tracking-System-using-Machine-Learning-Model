# Real-Time Classroom Attendance Tracking System using Machine Learning

This project is a **real-time classroom attendance tracking system** using a machine learning model for face recognition. The system is built with **Python**, **Flask**, **OpenCV**, and **face_recognition**, allowing automated attendance marking for registered students via webcam.

---

## Features

- Register new students with their photo and student ID.
- Real-time attendance tracking via webcam.
- Stores attendance in `attendance.csv`.
- Displays recognized students with their ID and counts on screen.
- Flash messages for feedback during registration and attendance recording.

---

## Folder Structure

Real-Time-Classroom-Attendance-Tracking-System/
│
├── src/
│ └── face_detection.py # Main Flask application
│
├── templates/
│ ├── index.html # Home page
│ └── register.html # Registration page
│
├── static/ # Optional static assets (CSS/images)
│ └── background.jpg # Background image
│
├── uploads/ # Uploaded student photos (auto-created)
├── .gitignore # Files/folders to ignore in Git
├── README.md
├── LICENSE
├── students.csv # Student database (auto-created)
├── face_encodings.pkl # Stored face encodings (auto-created)

---

> **Note:** `uploads/`, `students.csv`, and `face_encodings.pkl` are generated at runtime and should **not** be tracked in Git. `.gitignore` is included to ignore these files.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/Real-Time-Classroom-Attendance-Tracking-System.git
cd Real-Time-Classroom-Attendance-Tracking-System
2. Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate        # For Linux / Mac
venv\Scripts\activate           # For Windows


3. Install dependencies:

pip install -r requirements.txt


Example dependencies in requirements.txt:

flask
opencv-python
face_recognition
pandas
numpy
werkzeug
---

Usage

1. Run the Flask application:
python src/face_detection.py

2. Open your browser and go to:
http://127.0.0.1:5000/

3. Register students:
a) Fill in the name and student ID.
b) Upload a clear photo of the student.
c) Click "Register Student".

4. Start attendance recording:
a) Click the "Record Attendance" button.
b) The system will detect faces in real-time.
c) Attendance is saved in attendance.csv.

5. Stop attendance recording:
Press q in the webcam window.

Notes : Only students registered with their photos can be recognized.
        uploads/, students.csv, and face_encodings.pkl are created automatically during runtime.
        Make sure your webcam is working and accessible by OpenCV.
