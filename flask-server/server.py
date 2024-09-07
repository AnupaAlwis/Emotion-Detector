# # from flask import Flask

# # app = Flask(__name__)

# # @app.route("/members")
# # def members():
# #     return {"members": ["member1", "member2", "member3"]}

# # if __name__ == "__main__":
# #     app.run(debug=True)
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file:
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Process the image and get emotion
#         emotion = detect_emotion(file_path)  # Replace with your emotion detection function

#         return jsonify({'emotion': emotion})

# def detect_emotion(image_path):
#     # Implement your emotion detection logic here
#     # For example, load the model and predict emotion
#     print("Hello World")
#     return 'happy'  # Placeholder response

# if __name__ == '__main__':
#     app.run(debug=True)

#--------------------------------------------------------------------

# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array

# app = Flask(__name__)
# CORS(app)

# # Load the model
# face_classifier = cv2.CascadeClassifier(r'E:\ML_Projects\youtube\Emotion_Detection_CNN\haarcascade_frontalface_default.xml')
# classifier = load_model(r'E:\ML_Projects\youtube\Emotion_Detection_CNN\model.h5')

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# @app.route('/detect_emotion', methods=['POST'])
# def detect_emotion():
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         _, frame = cap.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray)

#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#             if np.sum([roi_gray]) != 0:
#                 roi = roi_gray.astype('float')/255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)

#                 prediction = classifier.predict(roi)[0]
#                 label = emotion_labels[prediction.argmax()]
                
#                 return jsonify({'emotion': label})
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#=------------------------------------------------------------------------------------------

# from flask import Flask, Response
# from flask_cors import CORS
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array

# app = Flask(__name__)
# CORS(app)

# # Load the model
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier = load_model('model.h5')

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# is_camera_active = False
# cap = None  # Global variable for the video capture

# def generate_frames():
#     global cap
    
#     while is_camera_active:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_classifier.detectMultiScale(gray)

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
#                 roi_gray = gray[y:y+h, x:x+w]
#                 roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#                 if np.sum([roi_gray]) != 0:
#                     roi = roi_gray.astype('float')/255.0
#                     roi = img_to_array(roi)
#                     roi = np.expand_dims(roi, axis=0)

#                     prediction = classifier.predict(roi)[0]
#                     label = emotion_labels[prediction.argmax()]
#                     label_position = (x, y)
#                     cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     global is_camera_active, cap
#     if not is_camera_active:
#         cap = cv2.VideoCapture(0)  # Initialize the camera when starting the video feed
#     is_camera_active = True
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/stop_video_feed', methods=['POST'])
# def stop_video_feed():
#     global is_camera_active, cap
#     is_camera_active = False
#     if cap:
#         cap.release()  # Release the camera
#     return '', 204

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#----------------------------------------------Camera works fine but the cancel issue presents -----------------
# from flask import Flask, Response, request
# from flask_cors import CORS
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array

# app = Flask(__name__)
# CORS(app)

# # Load the model
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier = load_model('model.h5')

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# is_camera_active = False
# cap = None  # Global variable for the video capture

# def generate_frames():
#     global cap
    
#     while is_camera_active:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_classifier.detectMultiScale(gray)

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
#                 roi_gray = gray[y:y+h, x:x+w]
#                 roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#                 if np.sum([roi_gray]) != 0:
#                     roi = roi_gray.astype('float')/255.0
#                     roi = img_to_array(roi)
#                     roi = np.expand_dims(roi, axis=0)

#                     prediction = classifier.predict(roi)[0]
#                     label = emotion_labels[prediction.argmax()]
#                     label_position = (x, y)
#                     cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n'
#                 b'Cache-Control: no-cache\r\n'
#                 b'Pragma: no-cache\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     global is_camera_active, cap
#     if not is_camera_active:
#         cap = cv2.VideoCapture(0)  # Initialize the camera when starting the video feed
#     is_camera_active = True
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/stop_video_feed', methods=['POST'])
# def stop_video_feed():
#     global is_camera_active, cap
#     is_camera_active = False
#     if cap:
#         cap.release()  # Release the camera
#     return '', 204

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#----------------------------------------------Camera works fine but the cancel issue presents -----------------
# from flask import Flask, Response, request
# from flask_cors import CORS
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array

# app = Flask(__name__)
# CORS(app)

# # Load the model
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier = load_model('model.h5')

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# cap = None  # Global variable for the video capture

# def generate_frames():
#     global cap
#     cap = cv2.VideoCapture(0)  # Reinitialize the camera when starting the video feed
    
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_classifier.detectMultiScale(gray)

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
#                 roi_gray = gray[y:y+h, x:x+w]
#                 roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#                 if np.sum([roi_gray]) != 0:
#                     roi = roi_gray.astype('float')/255.0
#                     roi = img_to_array(roi)
#                     roi = np.expand_dims(roi, axis=0)

#                     prediction = classifier.predict(roi)[0]
#                     label = emotion_labels[prediction.argmax()]
#                     label_position = (x, y)
#                     cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n'
#                 b'Cache-Control: no-cache\r\n'
#                 b'Pragma: no-cache\r\n\r\n' + frame + b'\r\n')
    
#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/stop_video_feed', methods=['POST'])
# def stop_video_feed():
#     global cap
#     if cap:
#         cap.release()  # Release the camera
#         cap = None
#     return '', 204

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#-----------------------------working camera not turning off----------------

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
classifier = load_model('../model.h5')

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

#----------------------------------------------Works fine but the camera can't be turned on again    

# from flask import Flask, Response, request
# from flask_cors import CORS
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import threading

# app = Flask(__name__)
# CORS(app)

# # Load the model
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier = load_model('model.h5')

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# stop_flag = threading.Event()  # Global flag to stop the video feed

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Initialize the camera

#     if not cap.isOpened():
#         print("Error: Could not open video source.")
#         return

#     try:
#         while not stop_flag.is_set():  # Check the stop flag
#             success, frame = cap.read()
#             if not success:
#                 print("Warning: Could not read frame from camera.")
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_classifier.detectMultiScale(gray)

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
#                 roi_gray = gray[y:y+h, x:x+w]
#                 roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#                 if np.sum([roi_gray]) != 0:
#                     roi = roi_gray.astype('float')/255.0
#                     roi = img_to_array(roi)
#                     roi = np.expand_dims(roi, axis=0)

#                     prediction = classifier.predict(roi)[0]
#                     label = emotion_labels[prediction.argmax()]
#                     label_position = (x, y)
#                     cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 print("Warning: Could not encode frame.")
#                 break

#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n'
#                 b'Cache-Control: no-cache\r\n'
#                 b'Pragma: no-cache\r\n\r\n' + frame + b'\r\n')
#     finally:
#         cap.release()
#         print("Camera released.")

# @app.route('/video_feed')
# def video_feed():
#     global stop_flag
#     stop_flag.clear()  # Reset the stop flag when the video feed starts
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/stop_video_feed', methods=['POST'])
# def stop_video_feed():
#     global stop_flag
#     stop_flag.set()  # Set the stop flag to stop the video feed
#     return '', 204

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
