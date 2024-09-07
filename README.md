# 🎭 Emotion Detection 

## 🚀 Project Overview

The **Emotion Detection Web App** is a real-time emotion recognition tool built using a **Haar Cascade Classifier** and a **Custom Convolutional Neural Network (CNN)** architecture. This web app uses the user's device camera to detect and classify emotions in real time, providing an interactive and dynamic experience.

By combining the power of computer vision and deep learning, this application can recognize several emotions. Whether you're feeling happy, sad, angry, or surprised, the app captures it all through live video feed analysis.

### 🎯 Key Features
- **Real-time Emotion Detection**: Utilizes the user's camera to detect emotions.
- **Custom-built CNN**: Emotion classification using a CNN built from scratch.
- **Haar Cascade Classifier**: Efficient face detection, ensuring the model focuses on the relevant region for emotion detection.

## 📷 How It Works

1. **Face Detection**: The app uses a **Haar Cascade Classifier** to detect faces in real time from the video feed.
2. **Emotion Classification**: Detected faces are passed through a **CNN model**, which identifies the user's emotion.
3. **Live Feedback**: The identified emotion is displayed in real-time on the user's screen, continuously updating as new frames are processed.

### Recognized Emotions:
- 😊 Happy
- 😡 Angry
- 😢 Sad
- 😱 Surprised
- 😐 Neutral
- 🤢 Disgust
- 😨 Fear

## 🛠️ Technologies Used

- **Flask**: Backend framework for API handling.
- **OpenCV**: Face detection using the Haar Cascade Classifier.
- **Keras & TensorFlow**: Custom CNN model for emotion classification.

## License

This project is licensed under the [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Contact

If you have any questions or suggestions, please feel free to reach out to us at [isurajithalwis@gmail.com](mailto:isurajithalwis@gmail.com).
