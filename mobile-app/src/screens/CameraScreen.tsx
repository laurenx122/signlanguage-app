import React, { useRef, useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { CameraView, useCameraPermissions, CameraType } from 'expo-camera';

const useScreenSize = () => {
  const { width, height } = Dimensions.get('window');
  const isSmallScreen = height < 600;
  
  return { isSmallScreen, screenWidth: width, screenHeight: height };
};

const CameraScreen: React.FC = () => {
  const cameraRef = useRef<CameraView>(null);
  const [prediction, setPrediction] = useState<string>('');
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('front');
  const { isSmallScreen, screenHeight } = useScreenSize();

  // Calculate 60% of screen height as a number
  const cameraHeight = screenHeight * 0.6;

  // Debug permissions
  useEffect(() => {
    console.log('Camera permission status:', permission);
  }, [permission]);

  const captureImage = async () => {
    if (!cameraRef.current) {
      console.log('Camera ref not available');
      return;
    }

    try {
      console.log('Attempting to capture image...');
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        base64: true,
        skipProcessing: true,
      });
      
      console.log('Image captured successfully');
      if (photo.base64) {
        processImage(photo.base64);
      } else {
        console.log('No base64 data in photo');
      }
    } catch (error) {
      console.error('Error capturing image:', error);
    }
  };

  const processImage = async (base64Image: string) => {
    try {
      console.log('Sending image to backend...');
      const response = await fetch('http://192.168.1.15:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Backend response:', result);
      setPrediction(result.prediction);
    } catch (error) {
      console.error('Error processing image:', error);
      setPrediction('Error: Could not process image');
    }
  };

  const toggleCameraFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>
          We need your permission to use the camera for sign language detection
        </Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionButtonText}>Grant Camera Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera View - 60% of screen */}
      <View style={[styles.cameraContainer, { height: cameraHeight }]}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing={facing}
          mode="picture"
        />
        
        {/* Camera Controls Overlay */}
        <View style={styles.cameraOverlay}>
          <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
            <Text style={styles.flipButtonText}>
              {facing === 'front' ? '📷 Back' : '📱 Front'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Bottom Section - 40% of screen */}
      <View style={styles.controlsContainer}>
        <TouchableOpacity style={styles.captureButton} onPress={captureImage}>
          <Text style={styles.captureText}>Capture Sign</Text>
        </TouchableOpacity>

        <View style={styles.predictionSection}>
          <Text style={styles.predictionLabel}>Detection Result:</Text>
          <View style={styles.predictionBox}>
            <Text style={styles.predictionText}>
              {prediction || 'Take a photo to detect sign language'}
            </Text>
          </View>
        </View>

        <View style={styles.instructions}>
          <Text style={styles.instructionsTitle}>How to use:</Text>
          <Text style={styles.instructionText}>1. Position your hand in the camera view</Text>
          <Text style={styles.instructionText}>2. Press "Capture Sign" to take a photo</Text>
          <Text style={styles.instructionText}>3. Wait for AI detection result</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { 
    flex: 1,
    backgroundColor: '#000',
  },
  cameraContainer: {
    width: '100%',
    position: 'relative',
  },
  camera: { 
    flex: 1,
  },
  cameraOverlay: {
    position: 'absolute',
    top: 20,
    right: 20,
  },
  flipButton: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#fff',
  },
  flipButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 14,
  },
  controlsContainer: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    padding: 20,
    justifyContent: 'space-between',
  },
  captureButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    alignSelf: 'center',
    marginBottom: 20,
    shadowColor: '#007AFF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  captureText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  predictionSection: {
    marginBottom: 20,
  },
  predictionLabel: {
    color: '#ccc',
    fontSize: 16,
    marginBottom: 10,
    textAlign: 'center',
  },
  predictionBox: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    padding: 15,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#333',
    minHeight: 60,
    justifyContent: 'center',
  },
  predictionText: {
    color: '#4CAF50',
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
  },
  instructions: {
    marginTop: 10,
  },
  instructionsTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  instructionText: {
    color: '#aaa',
    fontSize: 14,
    marginBottom: 5,
  },
  loadingText: {
    color: '#fff',
    fontSize: 18,
    textAlign: 'center',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    padding: 20,
  },
  permissionText: {
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 20,
    color: '#fff',
  },
  permissionButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 25,
  },
  permissionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default CameraScreen;