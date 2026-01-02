# Webcam Attendance System

A fast and efficient attendance management system using OpenCV face recognition and PyQt5 GUI.

## Features

- 🚀 **Fast Performance** - Optimized face recognition with OpenCV LBPH
- 📷 **Real-time Face Detection** - Automatic attendance marking via webcam
- 👥 **Student Management** - Add, edit, and manage student records
- 📊 **Attendance Reports** - View and export attendance data
- 📈 **Analytics Dashboard** - Visual attendance statistics
- 🔐 **Secure Login** - User authentication system
- ⚙️ **Configurable Settings** - Customize recognition parameters

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.7+
- OpenCV (opencv-python, opencv-contrib-python)
- PyQt5
- NumPy
- Pandas
- Matplotlib
- bcrypt
- cryptography

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd PythonProject1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Default Login

- **Username:** `admin`
- **Password:** `admin123`

## Usage

1. **Add Students**: 
   - Go to Students tab
   - Click "Add Student"
   - Enter student details and capture face (3 seconds)
   
2. **Mark Attendance**:
   - Go to Camera tab
   - Click "Start Camera"
   - Position face in front of camera
   - Wait 1.5 seconds for automatic marking
   - Camera auto-closes after successful marking

3. **View Reports**:
   - Check Attendance tab for daily records
   - Use Analytics tab for statistics
   - Export to CSV for external use

## Configuration

Adjust settings in the Settings tab:
- **Recognition Threshold**: 80 (lower = stricter, 0-200 range)
- **Detection Time**: 1.5 seconds (minimum time to mark attendance)
- **Cooldown Period**: 300 seconds (prevent duplicate marks)

## Performance Optimizations

- No image enhancement (maximum speed)
- Optimized face detection parameters
- Direct frame processing
- Fast LBPH face recognition

## License

This project is open source and available for educational purposes.

## Notes

- Ensure good lighting for best recognition accuracy
- Face the camera directly for optimal results
- Minimum face size: 100x100 pixels
- Database is stored locally as `attendance_system.db`
