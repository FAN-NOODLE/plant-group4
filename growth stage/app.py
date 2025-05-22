from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import webbrowser
import threading
import time

app = Flask(__name__)

# Load trained model
model = load_model('plant_growth_model.h5')

# Set camera ID (0 is default camera, 1 is external camera)
CAMERA_ID = 0  # Modify this value to select different camera

def preprocess_frame(frame):
    # Resize image and preprocess
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def generate_frames():
    # Use specified camera
    camera = cv2.VideoCapture(CAMERA_ID)
    
    # Set camera resolution (optional)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Unable to read camera feed, please check camera connection")
            break
        
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        
        # Predict
        prediction = model.predict(processed_frame)
        stage = "Seedling" if prediction[0][0] > prediction[0][1] else "Mature"
        confidence = max(prediction[0]) * 100
        
        # Add text to image
        cv2.putText(frame, f"Stage: {stage} ({confidence:.1f}%)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def open_browser():
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Open browser in new thread
    threading.Thread(target=open_browser).start()
    # Start Flask application
    app.run(debug=False) 