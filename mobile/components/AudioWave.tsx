import React from 'react';
import { View, StyleSheet } from 'react-native';
import LottieView from 'lottie-react-native';

export default function AudioWave({ isRecording }: { isRecording: boolean }) {
  return (
    <View style={styles.waveContainer}>
      {isRecording ? (
        <LottieView
          autoPlay
          loop
          source={require('../assets/audio-wave.json')} // You can find free audio wave JSONs on LottieFiles
          style={styles.wave}
        />
      ) : (
        <View style={styles.staticLine} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  waveContainer: { height: 100, justifyContent: 'center', alignItems: 'center' },
  wave: { width: '100%', height: 80 },
  staticLine: { width: '80%', height: 2, backgroundColor: '#ddd' }
});