import React, { useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Camera } from 'react-native-camera';

const CameraScreen: React.FC = () => {
  const cameraRef = useRef<Camera>(null);
  const [prediction, setPrediction] = useState<string>('');

  const captureImage = async () => {
    if (cameraRef.current) {
      const options = { quality: 0.5, base64: true };
      const data = await cameraRef.current.takePictureAsync(options);
      // Send to backend for processing
      processImage(data.base64);
    }
  };

  const processImage = async (base64Image: string) => {
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image }),
      });
      const result = await response.json();
      setPrediction(result.prediction);
    } catch (error) {
      console.error('Error processing image:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={styles.camera}
        type={Camera.Constants.Type.front}
      />
      <TouchableOpacity style={styles.captureButton} onPress={captureImage}>
        <Text style={styles.captureText}>Capture</Text>
      </TouchableOpacity>
      {prediction ? (
        <View style={styles.predictionContainer}>
          <Text style={styles.predictionText}>Prediction: {prediction}</Text>
        </View>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  captureButton: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 50,
  },
  captureText: { color: 'black' },
  predictionContainer: {
    position: 'absolute',
    top: 50,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 10,
    borderRadius: 10,
  },
  predictionText: { color: 'white', fontSize: 18 },
});

export default CameraScreen;