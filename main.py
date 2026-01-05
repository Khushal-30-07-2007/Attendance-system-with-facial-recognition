"""
Webcam Attendance System - OpenCV Only Version
No dlib dependency required! Uses OpenCV's built-in face recognition.

Requirements:
pip install opencv-python opencv-contrib-python PyQt5 pandas openpyxl pillow numpy bcrypt cryptography matplotlib
"""

import sys
import cv2
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import pickle
import bcrypt
import json
import os
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTableWidget, QTableWidgetItem, QMessageBox, 
                             QDialog, QFormLayout, QSpinBox,
                             QFileDialog, QTabWidget, QTextEdit, QCheckBox,
                             QGroupBox, QGridLayout, QDateEdit, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QDate
from PyQt5.QtGui import QImage, QPixmap, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ======================== CONFIGURATION ========================
class Config:
    DB_PATH = "attendance_system.db"
    CONFIG_FILE = "config.json"
    
    FACE_RECOGNITION_THRESHOLD = 80  # LBPH: lower confidence = better match (0-200 range, 80 is reasonable)
    MIN_FACE_DETECTION_TIME = 1.5    # Reduced for faster marking
    ATTENDANCE_COOLDOWN = 300
    MIN_FACE_SIZE = 100
    
    @classmethod
    def load_config(cls):
        if os.path.exists(cls.CONFIG_FILE):
            with open(cls.CONFIG_FILE, 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
    
    @classmethod
    def save_config(cls):
        config = {
            'FACE_RECOGNITION_THRESHOLD': cls.FACE_RECOGNITION_THRESHOLD,
            'MIN_FACE_DETECTION_TIME': cls.MIN_FACE_DETECTION_TIME,
            'ATTENDANCE_COOLDOWN': cls.ATTENDANCE_COOLDOWN
        }
        with open(cls.CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)


# ======================== DATABASE MANAGER ========================
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()
    
    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                face_encoding BLOB NOT NULL,
                registered_date TEXT DEFAULT CURRENT_TIMESTAMP,
                active INTEGER DEFAULT 1
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                FOREIGN KEY (student_id) REFERENCES students(id),
                UNIQUE(student_id, date)
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                role TEXT DEFAULT 'teacher',
                created_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        self.create_default_admin()
    
    def create_default_admin(self):
        try:
            password_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt())
            self.cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", password_hash, "admin")
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass
    
    def authenticate_user(self, username, password):
        self.cursor.execute(
            "SELECT id, password_hash, role FROM users WHERE username = ?",
            (username,)
        )
        result = self.cursor.fetchone()
        
        if result and bcrypt.checkpw(password.encode(), result[1]):
            return {'id': result[0], 'username': username, 'role': result[2]}
        return None
    
    def add_student(self, roll_number, name, email, phone, face_encoding):
        try:
            encoding_bytes = pickle.dumps(face_encoding)
            self.cursor.execute('''
                INSERT INTO students (roll_number, name, email, phone, face_encoding)
                VALUES (?, ?, ?, ?, ?)
            ''', (roll_number, name, email, phone, encoding_bytes))
            self.conn.commit()
            return True, "Student added successfully"
        except sqlite3.IntegrityError:
            return False, "Roll number already exists"
        except Exception as e:
            return False, str(e)
    
    def get_all_students(self):
        self.cursor.execute("SELECT id, roll_number, name, email, phone FROM students WHERE active=1")
        return self.cursor.fetchall()
    
    def get_student_by_id(self, student_id):
        self.cursor.execute("SELECT * FROM students WHERE id=?", (student_id,))
        return self.cursor.fetchone()
    
    def get_all_face_encodings(self):
        self.cursor.execute("SELECT id, roll_number, name, face_encoding FROM students WHERE active=1")
        results = self.cursor.fetchall()
        
        encodings = []
        for student_id, roll_number, name, encoding_blob in results:
            encoding = pickle.loads(encoding_blob)
            encodings.append({
                'id': student_id,
                'roll_number': roll_number,
                'name': name,
                'encoding': encoding
            })
        return encodings
    
    def mark_attendance(self, student_id, confidence):
        today = datetime.now().date().isoformat()  # Convert to string
        current_time = datetime.now().time().isoformat()  # Convert to string
        
        try:
            self.cursor.execute('''
                INSERT INTO attendance (student_id, date, time, confidence)
                VALUES (?, ?, ?, ?)
            ''', (student_id, today, current_time, confidence))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Already marked today
            print(f"⚠️ Attendance already marked for student ID {student_id} today")
            return False
        except Exception as e:
            print(f"❌ Error marking attendance: {e}")
            return False
    
    def get_attendance_by_date(self, date):
        # Convert date to string if it's a date object
        if hasattr(date, 'isoformat'):
            date = date.isoformat()
        
        self.cursor.execute('''
            SELECT s.roll_number, s.name, a.time, a.confidence
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time
        ''', (date,))
        return self.cursor.fetchall()
        return self.cursor.fetchall()
    
    def get_attendance_statistics(self, days=30):
        start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        end_date = datetime.now().date().isoformat()
        
        self.cursor.execute("SELECT COUNT(*) FROM students WHERE active=1")
        total_students = self.cursor.fetchone()[0]
        
        self.cursor.execute('''
            SELECT date, COUNT(DISTINCT student_id) as present_count
            FROM attendance
            WHERE date BETWEEN ? AND ?
            GROUP BY date
            ORDER BY date
        ''', (start_date, end_date))
        daily_attendance = self.cursor.fetchall()
        
        self.cursor.execute('''
            SELECT s.id, s.roll_number, s.name, COUNT(a.id) as days_present
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id 
                AND a.date BETWEEN ? AND ?
            WHERE s.active = 1
            GROUP BY s.id
        ''', (start_date, end_date))
        student_attendance = self.cursor.fetchall()
        
        return {
            'total_students': total_students,
            'daily_attendance': daily_attendance,
            'student_attendance': student_attendance,
            'date_range': (start_date, end_date)
        }
    
    def export_to_csv(self, filepath, start_date=None, end_date=None):
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).date().isoformat()
        elif hasattr(start_date, 'isoformat'):
            start_date = start_date.isoformat()
            
        if not end_date:
            end_date = datetime.now().date().isoformat()
        elif hasattr(end_date, 'isoformat'):
            end_date = end_date.isoformat()
        
        query = '''
            SELECT s.roll_number, s.name, a.date, a.time, a.confidence
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date BETWEEN ? AND ?
            ORDER BY a.date, a.time
        '''
        
        df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
        df.to_csv(filepath, index=False)
        return True
    
    def update_student(self, student_id, name=None, email=None, phone=None, face_encoding=None):
        updates = []
        params = []
        
        if name:
            updates.append("name = ?")
            params.append(name)
        if email:
            updates.append("email = ?")
            params.append(email)
        if phone:
            updates.append("phone = ?")
            params.append(phone)
        if face_encoding is not None:
            encoding_bytes = pickle.dumps(face_encoding)
            updates.append("face_encoding = ?")
            params.append(encoding_bytes)
        
        if updates:
            params.append(student_id)
            query = f"UPDATE students SET {', '.join(updates)} WHERE id = ?"
            self.cursor.execute(query, params)
            self.conn.commit()
            return True
        return False
    
    def delete_student(self, student_id):
        self.cursor.execute("UPDATE students SET active=0 WHERE id=?", (student_id,))
        self.conn.commit()
    
    def close(self):
        self.conn.close()


# ======================== OPENCV FACE RECOGNITION ENGINE ========================
class OpenCVFaceRecognizer:
    """Uses OpenCV's built-in LBPH Face Recognizer - No dlib needed!"""
    
    # Cache cascade classifier path (class variable - loaded once)
    _cascade_path = None
    
    @classmethod
    def _get_cascade_path(cls):
        if cls._cascade_path is None:
            cls._cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cls._cascade_path
    
    def __init__(self, db_manager):
        self.db = db_manager
        # Optimized LBPH parameters - balance of speed and accuracy
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,       # Reduced for speed
            neighbors=8,    # Standard value
            grid_x=8,       # Standard grid
            grid_y=8
        )
        # Use cached path for faster loading
        self.face_cascade = cv2.CascadeClassifier(self._get_cascade_path())
        
        self.known_face_metadata = []
        self.face_detection_tracker = {}
        self.last_attendance_time = {}
        
        self.is_trained = False
        self.load_and_train()
    
    def load_and_train(self):
        """Load all faces and train the recognizer"""
        students = self.db.get_all_face_encodings()
        
        if len(students) == 0:
            print("⚠️ No students found. Add students first.")
            self.is_trained = False
            return
        
        faces = []
        labels = []
        self.known_face_metadata = []
        
        for idx, student in enumerate(students):
            try:
                face_data = student['encoding']  # This should be the face image array
                # Ensure it's the right format
                if face_data is not None and face_data.shape == (200, 200):
                    faces.append(face_data)
                    labels.append(idx)
                    self.known_face_metadata.append({
                        'id': student['id'],
                        'roll_number': student['roll_number'],
                        'name': student['name']
                    })
                    print(f"✓ Loaded face for: {student['name']}")
                else:
                    print(f"✗ Invalid face data for: {student['name']}")
            except Exception as e:
                print(f"✗ Error loading face for {student['name']}: {e}")
        
        if len(faces) > 0:
            try:
                # Create new recognizer instance
                self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
                self.recognizer.train(faces, np.array(labels))
                self.is_trained = True
                print(f"✅ Successfully trained recognizer with {len(faces)} faces")
            except Exception as e:
                print(f"✗ Training failed: {e}")
                self.is_trained = False
        else:
            print("⚠️ No valid faces to train")
            self.is_trained = False
    
    
    def detect_faces(self, frame):
        """Detect faces - OPTIMIZED for speed"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Optimized parameters: larger scaleFactor = faster, fewer neighbors = faster
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,      # Larger = faster (was 1.1)
            minNeighbors=4,        # Fewer = faster (was 5)
            minSize=(Config.MIN_FACE_SIZE, Config.MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            faces = []
        return faces, gray
    
    def process_frame(self, frame):
        """Process frame - NO ENHANCEMENT for maximum speed"""
        faces, gray = self.detect_faces(frame)
        return faces, gray, frame
    
    def recognize_faces(self, faces, gray_frame, frame):
        """Recognize faces with voting system for accuracy"""
        recognized_faces = []
        current_time = time.time()
        
        if not self.is_trained:
            print("⚠️ Recognizer not trained. Add students first.")
            return recognized_faces
        
        for (x, y, w, h) in faces:
            # Extract face - no preprocessing for speed
            face_roi = gray_frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Recognize
            try:
                label, confidence = self.recognizer.predict(face_roi)
            except:
                continue
            
            # LBPH: lower confidence = better match (0 = perfect, typically 0-200)
            threshold = Config.FACE_RECOGNITION_THRESHOLD if Config.FACE_RECOGNITION_THRESHOLD > 0 else 80
            
            name = "Unknown"
            student_id = None
            can_mark = False
            
            # Check threshold and valid label
            if confidence < threshold and label < len(self.known_face_metadata) and label >= 0:
                metadata = self.known_face_metadata[label]
                name = metadata['name']
                student_id = metadata['id']
                
                print(f"✓ Recognized: {name} (confidence: {confidence:.1f}, threshold: {threshold})")
                
                # Track detection time
                if student_id not in self.face_detection_tracker:
                    self.face_detection_tracker[student_id] = current_time
                
                elapsed = current_time - self.face_detection_tracker[student_id]
                
                # Check if enough time has passed
                if elapsed >= Config.MIN_FACE_DETECTION_TIME:
                    if student_id not in self.last_attendance_time:
                        can_mark = True
                    else:
                        if (current_time - self.last_attendance_time[student_id]) >= Config.ATTENDANCE_COOLDOWN:
                            can_mark = True
            else:
                print(f"✗ Unknown (confidence: {confidence:.1f}, threshold: {threshold})")
            
            # Calculate confidence percentage (lower raw confidence = higher percentage)
            confidence_percent = max(0, min(100, (200 - confidence) / 2.0)) / 100.0
            
            recognized_faces.append({
                'name': name,
                'student_id': student_id,
                'confidence': confidence_percent,
                'location': (x, y, w, h),
                'can_mark_attendance': can_mark
            })
            
            if can_mark and student_id:
                if self.db.mark_attendance(student_id, confidence_percent):
                    self.last_attendance_time[student_id] = current_time
                    recognized_faces[-1]['attendance_marked'] = True
                    print(f"✅ Marked: {name}")
        
        return recognized_faces
    
    def register_new_face(self, image):
        """Register a new face from image - NO ENHANCEMENT for speed"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces, _ = self.detect_faces(image)
        
        if len(faces) == 0:
            return None, "No face detected"
        if len(faces) > 1:
            return None, "Multiple faces detected"
        
        x, y, w, h = faces[0]
        
        # Check face size
        if w < Config.MIN_FACE_SIZE or h < Config.MIN_FACE_SIZE:
            return None, "Face too small or too far"
        
        # Extract and resize face
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        
        return face_roi, "Face captured successfully"


# ======================== VIDEO CAPTURE THREAD ========================
class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, list)
    attendance_marked = pyqtSignal(str, str)  # student_name, roll_number
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.face_engine = None
        self.process_faces = True
        self.auto_close_on_mark = True  # New flag
    
    def set_face_engine(self, engine):
        self.face_engine = engine
    
    def set_auto_close(self, auto_close):
        self.auto_close_on_mark = auto_close
    
    def run(self):
        self.running = True
        # Optimized camera initialization for faster opening (Windows DirectShow backend)
        try:
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        except:
            cap = cv2.VideoCapture(self.camera_id)
        # Set properties before reading (faster initialization)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            recognized_faces = []
            
            if self.process_faces and self.face_engine:
                faces, gray, processed_frame = self.face_engine.process_frame(frame)
                recognized_faces = self.face_engine.recognize_faces(faces, gray, processed_frame)
                frame = processed_frame
                
                # Check if attendance was marked
                for face in recognized_faces:
                    if face.get('attendance_marked'):
                        student_name = face['name']
                        # Get student info
                        for metadata in self.face_engine.known_face_metadata:
                            if metadata['name'] == student_name:
                                self.attendance_marked.emit(student_name, metadata['roll_number'])
                                if self.auto_close_on_mark:
                                    self.running = False
                                break
            
            self.frame_ready.emit(frame, recognized_faces)
            frame_count += 1
            self.msleep(30)
        
        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()


# ======================== MAIN GUI ========================
class AttendanceSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Lazy loading - only initialize database and face engine after login
        self.db = None
        self.face_engine = None
        self.current_user = None
        self.video_thread = None
        self.detection_start_time = {}
        
        Config.load_config()
        self.init_login_ui()
    
    def init_login_ui(self):
        self.setWindowTitle("Attendance System - Login")
        self.setGeometry(100, 100, 400, 300)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(50, 50, 50, 50)
        
        title = QLabel("Attendance System")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.username_input.setMinimumHeight(40)
        layout.addWidget(self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(40)
        self.password_input.returnPressed.connect(self.login)
        layout.addWidget(self.password_input)
        
        login_btn = QPushButton("Login")
        login_btn.setMinimumHeight(40)
        login_btn.clicked.connect(self.login)
        layout.addWidget(login_btn)
        
        info_label = QLabel("Default: admin / admin123")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        central_widget.setLayout(layout)
    
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        user = self.db.authenticate_user(username, password)
        
        if user:
            self.current_user = user
            self.init_main_ui()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid credentials")
    
    def init_main_ui(self):
        self.setWindowTitle(f"Attendance System - {self.current_user['username']}")
        self.setGeometry(50, 50, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.tabs.addTab(self.create_camera_tab(), "📷 Camera")
        self.tabs.addTab(self.create_students_tab(), "👥 Students")
        self.tabs.addTab(self.create_attendance_tab(), "📊 Attendance")
        self.tabs.addTab(self.create_analytics_tab(), "📈 Analytics")
        self.tabs.addTab(self.create_settings_tab(), "⚙️ Settings")
        
        self.statusBar().showMessage("Ready - OpenCV Face Recognition")
    
    def create_camera_tab(self):
        tab = QWidget()
        layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        
        # Detection status label
        self.detection_status = QLabel("📷 Camera Ready")
        self.detection_status.setAlignment(Qt.AlignCenter)
        self.detection_status.setStyleSheet("""
            QLabel {
                background: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        left_layout.addWidget(self.detection_status)
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #ccc; background: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.camera_label)
        
        controls_layout = QHBoxLayout()
        self.start_camera_btn = QPushButton("▶ Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        controls_layout.addWidget(self.start_camera_btn)
        
        self.stop_camera_btn = QPushButton("⏸ Stop Camera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_camera_btn)
        
        test_btn = QPushButton("🧪 Test Recognition")
        test_btn.clicked.connect(self.test_recognition)
        test_btn.setToolTip("Capture a test image and see recognition results")
        controls_layout.addWidget(test_btn)
        
        left_layout.addLayout(controls_layout)
        
        # Auto-close checkbox
        auto_close_layout = QHBoxLayout()
        self.auto_close_checkbox = QCheckBox("Auto-close camera after attendance marked")
        self.auto_close_checkbox.setChecked(True)
        self.auto_close_checkbox.setToolTip("Automatically stop camera after successfully marking attendance")
        auto_close_layout.addWidget(self.auto_close_checkbox)
        auto_close_layout.addStretch()
        left_layout.addLayout(auto_close_layout)
        
        # Instructions
        instructions = QLabel(
            "📝 OPTIMIZED MODE Instructions:\n"
            "1. Click 'Start Camera' (smooth ~30 FPS)\n"
            "2. Face camera with good lighting\n"
            "3. System verifies 3 times (⏳ CHECKING)\n"
            "4. Once verified (✓ VERIFIED), wait 2.5s\n"
            "5. Progress bar fills → Attendance marked!\n"
            "6. Camera closes automatically\n\n"
            "✨ Fast recognition + High accuracy!"
        )
        instructions.setStyleSheet("""
            QLabel {
                background: #e7f3ff;
                padding: 10px;
                border-left: 4px solid #2196F3;
                font-size: 12px;
            }
        """)
        instructions.setWordWrap(True)
        left_layout.addWidget(instructions)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Today's Attendance:"))
        
        self.attendance_list = QTableWidget()
        self.attendance_list.setColumnCount(4)
        self.attendance_list.setHorizontalHeaderLabels(["Roll No", "Name", "Time", "Confidence"])
        self.attendance_list.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.attendance_list)
        
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        
        self.total_students_label = QLabel("Total Students: 0")
        self.present_today_label = QLabel("Present Today: 0")
        self.attendance_rate_label = QLabel("Attendance Rate: 0%")
        
        stats_layout.addWidget(self.total_students_label, 0, 0)
        stats_layout.addWidget(self.present_today_label, 0, 1)
        stats_layout.addWidget(self.attendance_rate_label, 1, 0, 1, 2)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_attendance_log)
        right_layout.addWidget(refresh_btn)
        
        layout.addLayout(left_layout, 2)
        layout.addLayout(right_layout, 1)
        
        tab.setLayout(layout)
        return tab
    
    def create_students_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        toolbar = QHBoxLayout()
        
        add_btn = QPushButton("➕ Add Student")
        add_btn.clicked.connect(self.add_student_dialog)
        toolbar.addWidget(add_btn)
        
        edit_btn = QPushButton("✏️ Edit")
        edit_btn.clicked.connect(self.edit_student_dialog)
        toolbar.addWidget(edit_btn)
        
        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.clicked.connect(self.delete_student)
        toolbar.addWidget(delete_btn)
        
        toolbar.addStretch()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_students_table)
        toolbar.addWidget(refresh_btn)
        
        retrain_btn = QPushButton("🔄 Retrain Recognizer")
        retrain_btn.clicked.connect(self.retrain_recognizer)
        retrain_btn.setToolTip("Reload and retrain face recognition model")
        toolbar.addWidget(retrain_btn)
        
        layout.addLayout(toolbar)
        
        self.students_table = QTableWidget()
        self.students_table.setColumnCount(4)
        self.students_table.setHorizontalHeaderLabels(["ID", "Roll Number", "Name", "Email"])
        self.students_table.horizontalHeader().setStretchLastSection(True)
        self.students_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.students_table)
        
        self.refresh_students_table()
        
        tab.setLayout(layout)
        return tab
    
    def create_attendance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Select Date:"))
        
        self.date_picker = QDateEdit()
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setCalendarPopup(True)
        date_layout.addWidget(self.date_picker)
        
        view_btn = QPushButton("View")
        view_btn.clicked.connect(self.view_attendance_by_date)
        date_layout.addWidget(view_btn)
        
        date_layout.addStretch()
        
        export_btn = QPushButton("📥 Export CSV")
        export_btn.clicked.connect(self.export_attendance)
        date_layout.addWidget(export_btn)
        
        layout.addLayout(date_layout)
        
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(4)
        self.attendance_table.setHorizontalHeaderLabels(["Roll No", "Name", "Time", "Confidence"])
        self.attendance_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.attendance_table)
        
        tab.setLayout(layout)
        return tab
    
    def create_analytics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Days:"))
        
        self.days_spinner = QSpinBox()
        self.days_spinner.setRange(7, 365)
        self.days_spinner.setValue(30)
        controls.addWidget(self.days_spinner)
        
        generate_btn = QPushButton("Generate Report")
        generate_btn.clicked.connect(self.generate_analytics)
        controls.addWidget(generate_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        self.analytics_figure = Figure(figsize=(12, 8))
        self.analytics_canvas = FigureCanvas(self.analytics_figure)
        layout.addWidget(self.analytics_canvas)
        
        self.analytics_text = QTextEdit()
        self.analytics_text.setReadOnly(True)
        self.analytics_text.setMaximumHeight(150)
        layout.addWidget(self.analytics_text)
        
        tab.setLayout(layout)
        return tab
    
    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        recognition_group = QGroupBox("Recognition Settings")
        recognition_layout = QFormLayout()
        
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(20, 80)
        self.threshold_spin.setValue(Config.FACE_RECOGNITION_THRESHOLD)
        self.threshold_spin.setToolTip("Lower = more strict (recommended: 30-50 for high accuracy)")
        recognition_layout.addRow("Recognition Threshold:", self.threshold_spin)
        
        self.detection_time_spin = QDoubleSpinBox()
        self.detection_time_spin.setRange(0.5, 10.0)
        self.detection_time_spin.setValue(Config.MIN_FACE_DETECTION_TIME)
        recognition_layout.addRow("Detection Time (s):", self.detection_time_spin)
        
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(60, 3600)
        self.cooldown_spin.setValue(Config.ATTENDANCE_COOLDOWN)
        recognition_layout.addRow("Cooldown (s):", self.cooldown_spin)
        
        recognition_group.setLayout(recognition_layout)
        layout.addWidget(recognition_group)
        
        info_label = QLabel(
            "ℹ️ OPTIMIZED HIGH PERFORMANCE MODE:\n\n"
            "⚡ Speed Optimizations:\n"
            "• Fast preprocessing (histogram equalization only)\n"
            "• 30 FPS camera display (smooth)\n"
            "• Process every 2nd frame (~15 FPS recognition)\n"
            "• 5 samples in 4 seconds (vs 8 in 6 seconds)\n"
            "• Reduced warm-up time\n\n"
            "🎯 Accuracy Features:\n"
            "• Voting System: 3 consistent matches required\n"
            "• Detection Time: 2.5 seconds verification\n"
            "• Smart face preprocessing\n\n"
            "Recognition Threshold (20-80):\n"
            "• 40-50: Balanced (recommended)\n"
            "• 50-60: Faster, slightly less strict\n\n"
            "💡 Best of both worlds:\n"
            "Fast, smooth camera + accurate recognition!"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #0066cc; padding: 15px; background: #e6f3ff; border-radius: 5px;")
        layout.addWidget(info_label)
        
        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def start_camera(self):
        if self.video_thread is None or not self.video_thread.running:
            self.video_thread = VideoCaptureThread(camera_id=0)
            self.video_thread.set_face_engine(self.face_engine)
            self.video_thread.set_auto_close(self.auto_close_checkbox.isChecked())
            self.video_thread.frame_ready.connect(self.display_frame)
            self.video_thread.attendance_marked.connect(self.on_attendance_marked)
            self.video_thread.start()
            
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.detection_status.setText("🔍 Detecting faces...")
            self.detection_status.setStyleSheet("""
                QLabel {
                    background: #fff3cd;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                    color: #856404;
                }
            """)
            self.statusBar().showMessage("Camera started - Position your face for attendance")
            
            self.refresh_timer = QTimer()
            self.refresh_timer.timeout.connect(self.refresh_attendance_log)
            self.refresh_timer.start(5000)
            
            # Track detection time
            self.detection_start_time = {}
    
    def on_attendance_marked(self, student_name, roll_number):
        """Called when attendance is successfully marked"""
        # Update status
        self.detection_status.setText(f"✅ Attendance Marked: {student_name}")
        self.detection_status.setStyleSheet("""
            QLabel {
                background: #d4edda;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                color: #155724;
            }
        """)
        
        # Show success message
        QMessageBox.information(
            self,
            "✅ Attendance Marked",
            f"Attendance successfully marked!\n\n"
            f"Name: {student_name}\n"
            f"Roll Number: {roll_number}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"Camera will now close automatically."
        )
        
        # Refresh attendance list
        self.refresh_attendance_log()
        
        # Stop camera
        self.stop_camera()
        
        print(f"\n{'='*50}")
        print(f"✅ ATTENDANCE MARKED SUCCESSFULLY")
        print(f"Student: {student_name} ({roll_number})")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}\n")
    
    def stop_camera(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.camera_label.clear()
            self.detection_status.setText("📷 Camera Ready")
            self.detection_status.setStyleSheet("""
                QLabel {
                    background: #f0f0f0;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            self.statusBar().showMessage("Camera stopped")
            
            if hasattr(self, 'refresh_timer'):
                self.refresh_timer.stop()
            
            if hasattr(self, 'detection_start_time'):
                self.detection_start_time.clear()
    
    def test_recognition(self):
        """Test face recognition with current setup"""
        if not self.face_engine.is_trained:
            QMessageBox.warning(
                self, 
                "Not Trained", 
                "Face recognizer is not trained!\n\nPlease add students first, then use 'Retrain Recognizer' button in Students tab."
            )
            return
        
        # Optimized camera initialization
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except:
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Quick warm up (reduced)
        cap.read()
        
        QMessageBox.information(self, "Test Recognition", "Position your face in front of camera.\nTest will run for 3 seconds.")
        
        results = []
        start_time = time.time()
        
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if ret:
                faces, gray, processed = self.face_engine.process_frame(frame)
                recognized = self.face_engine.recognize_faces(faces, gray, processed)
                
                if recognized:
                    results.append(recognized[0])
                
                # Show preview
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imshow("Test Recognition", frame)
                cv2.waitKey(1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show results
        if results:
            # Get most common result
            names = [r['name'] for r in results]
            most_common = max(set(names), key=names.count)
            confidence = np.mean([r['confidence'] for r in results if r['name'] == most_common])
            
            msg = f"🎯 Recognition Test Results:\n\n"
            msg += f"Detected: {most_common}\n"
            msg += f"Average Confidence: {confidence:.2%}\n"
            msg += f"Detections: {names.count(most_common)}/{len(results)}\n\n"
            msg += f"Check console output for detailed recognition values."
            
            QMessageBox.information(self, "Test Results", msg)
        else:
            QMessageBox.warning(self, "Test Results", "No faces detected during test period!")
    
    def display_frame(self, frame, recognized_faces):
        # Track detection progress
        current_time = time.time()
        detecting_someone = False
        
        for face in recognized_faces:
            x, y, w, h = face['location']
            
            # Determine color based on recognition
            if face['student_id']:
                if face.get('attendance_marked'):
                    color = (0, 255, 0)  # Green - marked
                else:
                    color = (0, 165, 255)  # Orange - recognized
                detecting_someone = True
                
                # Track time for detection status
                student_id = face['student_id']
                if student_id not in self.detection_start_time:
                    self.detection_start_time[student_id] = current_time
                
                elapsed = current_time - self.detection_start_time[student_id]
                remaining = max(0, Config.MIN_FACE_DETECTION_TIME - elapsed)
                
                # Update status with countdown
                if remaining > 0 and not face.get('attendance_marked'):
                    self.detection_status.setText(f"👤 {face['name']} - {remaining:.1f}s")
                    self.detection_status.setStyleSheet("""
                        QLabel {
                            background: #cfe2ff;
                            padding: 10px;
                            border-radius: 5px;
                            font-size: 14px;
                            font-weight: bold;
                            color: #084298;
                        }
                    """)
                elif face.get('attendance_marked'):
                    self.detection_status.setText(f"✅ Marked: {face['name']}")
                    self.detection_status.setStyleSheet("""
                        QLabel {
                            background: #d4edda;
                            padding: 10px;
                            border-radius: 5px;
                            font-size: 14px;
                            font-weight: bold;
                            color: #155724;
                        }
                    """)
            else:
                color = (0, 0, 255)  # Red - unknown
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Create label with confidence and status
            conf_percent = face['confidence'] * 100
            label = f"{face['name']} ({conf_percent:.0f}%)"
            if face.get('attendance_marked'):
                label += " ✓"
            
            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.rectangle(frame, (x, y-label_height-15), (x+label_width+15, y), color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(frame, label, (x + 7, y - 7), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            
            # Draw progress indicator if detecting
            if face['student_id'] and not face.get('attendance_marked'):
                elapsed = current_time - self.detection_start_time.get(face['student_id'], current_time)
                progress = min(1.0, elapsed / Config.MIN_FACE_DETECTION_TIME)
                
                # Draw progress bar
                bar_width = w
                bar_height = 10
                bar_x = x
                bar_y = y + h + 5
                
                # Background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), cv2.FILLED)
                
                # Progress
                progress_width = int(bar_width * progress)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), cv2.FILLED)
        
        # Update status if no faces detected
        if not detecting_someone and hasattr(self, 'detection_start_time'):
            self.detection_start_time.clear()
            if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.running:
                self.detection_status.setText("🔍 Detecting faces... Position yourself in frame")
                self.detection_status.setStyleSheet("""
                    QLabel {
                        background: #fff3cd;
                        padding: 10px;
                        border-radius: 5px;
                        font-size: 14px;
                        font-weight: bold;
                        color: #856404;
                    }
                """)
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), 
                                      Qt.KeepAspectRatio, 
                                      Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
    
    def add_student_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Student")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        roll_input = QLineEdit()
        name_input = QLineEdit()
        email_input = QLineEdit()
        phone_input = QLineEdit()
        
        layout.addRow("Roll Number:", roll_input)
        layout.addRow("Name:", name_input)
        layout.addRow("Email:", email_input)
        layout.addRow("Phone:", phone_input)
        
        face_layout = QHBoxLayout()
        capture_btn = QPushButton("📷 Capture Face")
        capture_btn.setToolTip("Captures 5 samples over 4 seconds (FAST).\nStay still with good lighting.\nPress ESC to cancel.")
        face_status = QLabel("No face captured")
        face_layout.addWidget(capture_btn)
        face_layout.addWidget(face_status)
        layout.addRow("Face:", face_layout)
        
        face_encoding = [None]
        
        def capture_face():
            try:
                # Optimized camera initialization
                try:
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                except:
                    cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Quick warm up (reduced)
                cap.read()
                
                best_face = None
                best_size = 0
                
                start_time = time.time()
                while time.time() - start_time < 3:  # Reduced to 3 seconds
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_engine.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    # Get best face
                    for (x, y, w, h) in faces:
                        if w * h > best_size:
                            best_size = w * h
                            best_face = gray[y:y+h, x:x+w]
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    remaining = 3 - int(time.time() - start_time)
                    cv2.putText(frame, f"{remaining}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Capture Face", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                
                cv2.destroyAllWindows()
                cap.release()
                
                if best_face is not None:
                    best_face = cv2.resize(best_face, (200, 200))
                    face_encoding[0] = best_face
                    face_status.setText(f"✓ Captured ({int(np.sqrt(best_size))}px)")
                    face_status.setStyleSheet("color: green;")
                else:
                    face_status.setText("✗ No face")
                    face_status.setStyleSheet("color: red;")
            
            except Exception as e:
                print(f"❌ Error: {e}")
                cv2.destroyAllWindows()
                if 'cap' in locals():
                    cap.release()
                face_status.setText(f"✗ Error")
                face_status.setStyleSheet("color: red;")
        
        capture_btn.clicked.connect(capture_face)
        
        buttons = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)
        layout.addRow(buttons)
        
        def save_student():
            if not all([roll_input.text(), name_input.text(), face_encoding[0] is not None]):
                QMessageBox.warning(dialog, "Error", "Fill all fields and capture face")
                return
            
            success, message = self.db.add_student(
                roll_input.text(),
                name_input.text(),
                email_input.text(),
                phone_input.text(),
                face_encoding[0]
            )
            
            if success:
                self.face_engine.load_and_train()  # Retrain with new face
                self.refresh_students_table()
                QMessageBox.information(dialog, "Success", message)
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Error", message)
        
        save_btn.clicked.connect(save_student)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def edit_student_dialog(self):
        selected = self.students_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Select a student")
            return
        
        row = self.students_table.currentRow()
        student_id = int(self.students_table.item(row, 0).text())
        student = self.db.get_student_by_id(student_id)
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Student")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        name_input = QLineEdit(student[2])
        email_input = QLineEdit(student[3] or "")
        phone_input = QLineEdit(student[4] or "")
        
        layout.addRow("Name:", name_input)
        layout.addRow("Email:", email_input)
        layout.addRow("Phone:", phone_input)
        
        face_layout = QHBoxLayout()
        recapture_btn = QPushButton("📷 Re-capture Face")
        recapture_btn.setToolTip("Captures 5 new samples over 4 seconds (FAST).\nStay still with good lighting.\nPress ESC to cancel.")
        face_status = QLabel("Current face")
        face_layout.addWidget(recapture_btn)
        face_layout.addWidget(face_status)
        layout.addRow("Face:", face_layout)
        
        new_face_encoding = [None]
        
        def recapture_face():
            try:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                for _ in range(3):
                    cap.read()
                
                best_face = None
                best_size = 0
                
                start_time = time.time()
                while time.time() - start_time < 3:  # Reduced to 3 seconds
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_engine.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        if w * h > best_size:
                            best_size = w * h
                            best_face = gray[y:y+h, x:x+w]
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    remaining = 3 - int(time.time() - start_time)
                    cv2.putText(frame, f"{remaining}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Recapture Face", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                
                cv2.destroyAllWindows()
                cap.release()
                
                if best_face is not None:
                    best_face = cv2.resize(best_face, (200, 200))
                    new_face_encoding[0] = best_face
                    face_status.setText(f"✓ Captured ({int(np.sqrt(best_size))}px)")
                    face_status.setStyleSheet("color: green;")
                else:
                    face_status.setText("✗ No face")
                    face_status.setStyleSheet("color: red;")
            
            except Exception as e:
                print(f"❌ Error: {e}")
                cv2.destroyAllWindows()
                if 'cap' in locals():
                    cap.release()
                face_status.setText(f"✗ Error")
                face_status.setStyleSheet("color: red;")
        
        recapture_btn.clicked.connect(recapture_face)
        
        buttons = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)
        layout.addRow(buttons)
        
        def update_student():
            success = self.db.update_student(
                student_id,
                name=name_input.text(),
                email=email_input.text(),
                phone=phone_input.text(),
                face_encoding=new_face_encoding[0]
            )
            
            if success:
                self.face_engine.load_and_train()
                self.refresh_students_table()
                QMessageBox.information(dialog, "Success", "Student updated")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Error", "Update failed")
        
        save_btn.clicked.connect(update_student)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def delete_student(self):
        selected = self.students_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Select a student")
            return
        
        row = self.students_table.currentRow()
        student_id = int(self.students_table.item(row, 0).text())
        
        reply = QMessageBox.question(self, "Confirm", "Delete this student?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.db.delete_student(student_id)
            self.face_engine.load_and_train()
            self.refresh_students_table()
            QMessageBox.information(self, "Success", "Student deleted")
    
    def refresh_students_table(self):
        students = self.db.get_all_students()
        self.students_table.setRowCount(len(students))
        
        for i, student in enumerate(students):
            for j, value in enumerate(student):
                self.students_table.setItem(i, j, QTableWidgetItem(str(value)))
        
        self.total_students_label.setText(f"Total Students: {len(students)}")
    
    def retrain_recognizer(self):
        """Manually retrain the face recognizer"""
        reply = QMessageBox.question(
            self, 
            "Retrain Recognizer", 
            "This will reload all student faces and retrain the recognition model.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            print("\n" + "="*50)
            print("🔄 Retraining face recognizer...")
            print("="*50)
            self.face_engine.load_and_train()
            print("="*50 + "\n")
            
            if self.face_engine.is_trained:
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Recognizer retrained successfully!\n\nLoaded {len(self.face_engine.known_face_metadata)} faces."
                )
            else:
                QMessageBox.warning(
                    self, 
                    "Failed", 
                    "Failed to train recognizer. Check console for errors."
                )
    
    def refresh_attendance_log(self):
        today = datetime.now().date().isoformat()
        records = self.db.get_attendance_by_date(today)
        
        self.attendance_list.setRowCount(len(records))
        
        for i, record in enumerate(records):
            for j, value in enumerate(record):
                if j == 3:  # Confidence
                    value = f"{value:.2%}"
                self.attendance_list.setItem(i, j, QTableWidgetItem(str(value)))
        
        students = self.db.get_all_students()
        total = len(students)
        present = len(records)
        rate = (present / total * 100) if total > 0 else 0
        
        self.present_today_label.setText(f"Present Today: {present}")
        self.attendance_rate_label.setText(f"Attendance Rate: {rate:.1f}%")
    
    def view_attendance_by_date(self):
        selected_date = self.date_picker.date().toPyDate().isoformat()
        records = self.db.get_attendance_by_date(selected_date)
        
        self.attendance_table.setRowCount(len(records))
        
        for i, record in enumerate(records):
            for j, value in enumerate(record):
                if j == 3:
                    value = f"{value:.2%}"
                self.attendance_table.setItem(i, j, QTableWidgetItem(str(value)))
    
    def export_attendance(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        
        if filepath:
            try:
                self.db.export_to_csv(filepath)
                QMessageBox.information(self, "Success", "Exported successfully")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Export failed: {e}")
    
    def generate_analytics(self):
        days = self.days_spinner.value()
        stats = self.db.get_attendance_statistics(days)
        
        self.analytics_figure.clear()
        
        # Daily attendance chart
        ax1 = self.analytics_figure.add_subplot(2, 1, 1)
        dates = [d[0] for d in stats['daily_attendance']]
        counts = [d[1] for d in stats['daily_attendance']]
        
        if len(dates) > 0:
            ax1.plot(dates, counts, marker='o')
            ax1.set_title('Daily Attendance')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Students Present')
            ax1.grid(True)
        
        # Student attendance bar chart
        ax2 = self.analytics_figure.add_subplot(2, 1, 2)
        student_names = [s[2][:15] for s in stats['student_attendance'][:10]]
        attendance_days = [s[3] for s in stats['student_attendance'][:10]]
        
        if len(student_names) > 0:
            ax2.barh(student_names, attendance_days)
            ax2.set_title('Top 10 Students Attendance')
            ax2.set_xlabel('Days Present')
        
        self.analytics_figure.tight_layout()
        self.analytics_canvas.draw()
        
        # Generate text report
        report = f"Attendance Report ({days} days)\n"
        report += f"Total Students: {stats['total_students']}\n"
        if len(counts) > 0:
            report += f"Average Daily Attendance: {np.mean(counts):.1f}\n"
        
        low_attendance = [s for s in stats['student_attendance'] if s[3] < days * 0.75]
        if low_attendance:
            report += f"\n⚠️ {len(low_attendance)} students with <75% attendance"
        
        self.analytics_text.setText(report)
    
    def save_settings(self):
        Config.FACE_RECOGNITION_THRESHOLD = self.threshold_spin.value()
        Config.MIN_FACE_DETECTION_TIME = self.detection_time_spin.value()
        Config.ATTENDANCE_COOLDOWN = self.cooldown_spin.value()
        
        Config.save_config()
        QMessageBox.information(self, "Success", "Settings saved. Restart for changes to take effect.")
    
    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if self.db:
            self.db.close()
        event.accept()


# ======================== MAIN ========================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = AttendanceSystemGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

