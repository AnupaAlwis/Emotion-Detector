import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import cameraIcon from './camera_5844892-removebg-preview.png';

function App() {
  const [isCameraActive, setIsCameraActive] = useState(false);

  const handleCameraToggle = async () => {
    if (isCameraActive) {
      await axios.post('http://localhost:5000/stop_video_feed');
      setIsCameraActive(false);
    } else {
      await new Promise((resolve) => setTimeout(resolve, 500));
      setIsCameraActive(true);
    }
  };

  return (
    <div className="App">
      <h1>Real-Time Emotion Detector</h1>
      
      {!isCameraActive ? (
        <button onClick={handleCameraToggle} className="camera-button center">
          <img src={cameraIcon} alt="Open Camera" className="camera-icon" />
        </button>
      ) : (
        <div className="video-container">
          <div className="video-feed">
            <img src="http://localhost:5000/video_feed" alt="Camera Feed" />
          </div>
          <button onClick={handleCameraToggle} className="camera-button center">
            <img src={cameraIcon} alt="Open Camera" className="camera-icon" />
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
