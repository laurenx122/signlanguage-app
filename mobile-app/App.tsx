// App.tsx - Simple version without animations first
import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import CameraScreen from './src/screens/CameraScreen';

export default function App() {
  const [showCamera, setShowCamera] = useState(false);

  if (!showCamera) {
    return (
      <View style={styles.introContainer}>
        <StatusBar style="light" />
        <View style={styles.introContent}>
          <Text style={styles.title}>Sign Language Recognition</Text>
          <Text style={styles.subtitle}>
            AI-powered sign language detection
          </Text>
          <TouchableOpacity style={styles.startButton} onPress={() => setShowCamera(true)}>
            <Text style={styles.startButtonText}>Start Recognition</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      <CameraScreen />
    </View>
  );
}

const styles = StyleSheet.create({
  introContainer: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  introContent: {
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 16,
  },
  subtitle: {
    fontSize: 16,
    color: '#ccc',
    textAlign: 'center',
    marginBottom: 40,
  },
  startButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
  },
  startButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
});