from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import os
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flash messages

# Directories for image uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Persistent storage files
STUDENT_DATA_FILE = 'students.csv'
FACE_ENCODINGS_FILE = 'face_encodings.pkl'

# Known face encodings and student details
known_face_encodings = []
known_face_names = []
known_face_ids = []

attendance = {}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to mark attendance
def mark_attendance(name, student_id):
    if name not in attendance:
        now = datetime.now()
        time_string = now.strftime('%Y-%m-%d %H:%M:%S')
        attendance[name] = time_string
        df = pd.DataFrame([[name, student_id, time_string]], columns=['Name', 'Student_ID', 'Date_Time'])
        
        # Check if CSV file exists, and include headers only if it's a new file
        if not os.path.exists("attendance.csv"):
            df.to_csv("attendance.csv", mode='w', index=False, header=True)
        else:
            df.to_csv("attendance.csv", mode='a', index=False, header=False)
        
        flash(f"Attendance marked for {name} (ID: {student_id}) at {time_string}!")
        print(f"{name} (ID: {student_id}) marked present at {time_string}")

# Load student data and face encodings when the app starts
def load_student_data():
    global known_face_encodings, known_face_names, known_face_ids
    # Load student names and IDs
    if os.path.exists(STUDENT_DATA_FILE):
        student_df = pd.read_csv(STUDENT_DATA_FILE)
        known_face_names = student_df['Name'].tolist()
        known_face_ids = student_df['Student_ID'].tolist()
    else:
        known_face_names = []
        known_face_ids = []

    # Load face encodings
    if os.path.exists(FACE_ENCODINGS_FILE):
        with open(FACE_ENCODINGS_FILE, 'rb') as f:
            known_face_encodings = pickle.load(f)
    else:
        known_face_encodings = []

# Save student data and face encodings
def save_student_data():
    # Save student names and IDs
    student_df = pd.DataFrame({'Name': known_face_names, 'Student_ID': known_face_ids})
    student_df.to_csv(STUDENT_DATA_FILE, index=False)

    # Save face encodings
    with open(FACE_ENCODINGS_FILE, 'wb') as f:
        pickle.dump(known_face_encodings, f)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle student registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        student_id = request.form['student_id']
        
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash("No file uploaded. Please try again.")
            return redirect(request.url)
        
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Load the uploaded image and extract face encoding
            known_image = face_recognition.load_image_file(file_path)
            known_face_encoding = face_recognition.face_encodings(known_image)
            
            if known_face_encoding:
                # Append to in-memory lists
                known_face_encodings.append(known_face_encoding[0])
                known_face_names.append(name)
                known_face_ids.append(student_id)

                # Save to persistent storage
                save_student_data()

                flash(f"Student {name} (ID: {student_id}) has been successfully registered!")
                return redirect(url_for('home'))
            else:
                flash("No face found in the image. Please upload a valid image.")
                return redirect(request.url)
        else:
            flash("Invalid file type. Please upload a .jpg, .jpeg, or .png file.")
            return redirect(request.url)

    return render_template('register.html')

# Route to start attendance capture
@app.route('/start-attendance')
def start_attendance():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            student_id = ""

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin() if matches else None
            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]
                student_id = known_face_ids[best_match_index]
                mark_attendance(name, student_id)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({student_id})", (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Attendance Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    flash("Attendance recording has stopped.")
    return redirect(url_for('home'))

if __name__ == "__main__":
    # Load previously registered student data
    load_student_data()
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)  
