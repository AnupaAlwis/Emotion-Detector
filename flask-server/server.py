
from flask import Flask, Response, request
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app)

# Load the model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('./model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def generate_frames():
    
    cap = cv2.VideoCapture(0)  # Initialize the camera
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: Could not read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x+10, y-10)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Warning: Could not encode frame.")
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Cache-Control: no-cache\r\n'
                b'Pragma: no-cache\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()
        print("Camera released.")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_feed', methods=['POST'])
def stop_video_feed():
    # This endpoint stops the video feed by not calling generate_frames again
    
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

