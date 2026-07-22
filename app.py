import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
from ultralytics import YOLO
import pyttsx3
import time
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLOv11 model for object detection
model = YOLO("yolo11l.pt")
names = model.model.names

# Load the trained classification model for money detection
money_model = YOLO('best_s.pt')
money_class_names = ["10", "100", "20", "200", "2000", "50", "500"]

# Global variables for detected objects and last spoken time
detected_objects = set()
last_spoken_time = {}
audio_lock = threading.Lock()  # Lock to synchronize audio feedback

manual_audio_trigger = False  # Flag to trigger audio manually
continuous_audio_active = False  # Flag for continuous audio feedback
detection_active = False  # Flag for active detection state

# Global variable to store the latest detected currency
latest_currency = None


# Initialize pyttsx3 engine
def speak_pyttsx3(text):
    """Speak the given text using pyttsx3."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in speak_pyttsx3: {e}")


@app.route('/trigger_audio', methods=['POST'])
def trigger_audio():
    global manual_audio_trigger
    manual_audio_trigger = True
    return 'Audio Triggered', 200


@app.route('/toggle_continuous_audio', methods=['POST'])
def toggle_continuous_audio():
    global continuous_audio_active
    data = request.get_json()
    continuous_audio_active = data.get('active', False)
    return 'Continuous Audio Toggled', 200


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    data = request.get_json()
    detection_active = data.get('active', False)
    return 'Detection state updated', 200


def audio_feedback_thread():
    """Thread to provide manual and continuous audio feedback for detected objects."""
    global detected_objects, last_spoken_time, manual_audio_trigger, continuous_audio_active
    while True:
        try:
            current_time = time.time()
            with audio_lock:
                objects_to_speak = detected_objects.copy()
                for obj in objects_to_speak:
                    # Check if continuous audio is active or manual trigger is pressed
                    if continuous_audio_active or manual_audio_trigger:
                        manual_audio_trigger = False  # Reset manual trigger
                        message = f"I see {obj}"
                        print(f"Speaking: {message}")
                        speak_pyttsx3(message)
                        last_spoken_time[obj] = current_time
            time.sleep(1)
        except Exception as e:
            print(f"Error in audio_feedback_thread: {e}")


# Start the audio feedback thread
threading.Thread(target=audio_feedback_thread, daemon=True).start()


def detect_objects_from_webcam():
    global detected_objects
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        with audio_lock:
            detected_objects = set()
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                    c = names[class_id]
                    detected_objects.add(c)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_objects_from_external_webcam():
    global detected_objects
    # Change the index to 1 for external webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open external webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        with audio_lock:
            detected_objects = set()
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                    c = names[class_id]
                    detected_objects.add(c)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def classify_frame(frame):
    try:
        results = money_model(frame)
        # Get confidence of the top prediction
        top_conf = results[0].probs.top1conf.item()
        # Only return prediction if confidence is above threshold (e.g., 0.7)
        if top_conf > 0.7:
            pred_idx = results[0].probs.top1
            predicted_label = money_class_names[pred_idx]
            return predicted_label
        return None
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None


def generate_money_frames():
    global latest_currency, detection_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Classify the frame
        currency = classify_frame(frame)
        latest_currency = currency  # Update the latest currency

        # Display results on the frame
        if currency:
            # Change color and thickness based on detection state
            color = (0, 255, 0) if detection_active else (0, 0, 255)
            thickness = 4 if detection_active else 2
            text = f"{currency} Rupees"

            # Add a border around the entire frame when detection is active
            if detection_active:
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 10)

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
        else:
            # Clear the latest currency if nothing detected
            latest_currency = None
            if detection_active:
                cv2.putText(frame, "No currency detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_webcam')
def start_webcam():
    return render_template('webcam.html')


@app.route('/start_external_webcam')
def start_external_webcam():
    return render_template('externalweb.html')


@app.route('/start_money_detection')
def start_money_detection():
    return render_template('money_detection.html')


@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/external_webcam_feed')
def external_webcam_feed():
    return Response(detect_objects_from_external_webcam(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/money_feed')
def money_feed():
    return Response(generate_money_frames(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_currency')
def detect_currency():
    return jsonify({'currency': latest_currency if latest_currency else ''})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files and 'photo' not in request.files:
        return "No file part", 400

    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            # Save the photo
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
            photo.save(file_path)

            # Process the photo
            img = cv2.imread(file_path)
            img = cv2.resize(img, (1020, 600))
            results = model.track(img, persist=True)

            # Draw bounding boxes
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                for box, class_id in zip(boxes, class_ids):
                    c = names[class_id]
                    x1, y1, x2, y2 = box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save the processed image
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + photo.filename)
            cv2.imwrite(processed_path, img)
            return redirect(url_for('show_photo', filename='processed_' + photo.filename))

    # Handle video upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('play_video', filename=file.filename))

    return "No selected file", 400


@app.route('/play_video/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)


@app.route('/show_photo/<filename>')
def show_photo(filename):
    return render_template('show_photo.html', filename=filename)


@app.route('/video_feed/<filename>')
def video_feed(filename):
    # Open the video file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(video_path)

    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Resize the frame
            frame = cv2.resize(frame, (1020, 600))

            # Perform object detection using YOLO
            results = model.track(frame, persist=True)

            # Draw bounding boxes and labels on the frame
            with audio_lock:
                detected_objects_local = set()
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                        c = names[class_id]
                        detected_objects_local.add(c)
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/photo/<filename>')
def photo(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
