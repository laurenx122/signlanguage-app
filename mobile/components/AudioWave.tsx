import React from "react";
import { View, StyleSheet } from "react-native";
import LottieView from "lottie-react-native";

export default function AudioWave({ isRecording }: { isRecording: boolean }) {
  return (
    <View style={styles.container}>
      {isRecording ? (
        <LottieView
          autoPlay
          loop
          source={require("../assets/audio-wave.json")}
          style={styles.wave}
        />
      ) : (
        <View style={styles.staticLine} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
    height: 120,
    justifyContent: "center",
    alignItems: "center",
  },

  wave: {
    width: 120,
    height: 60,
  },

  staticLine: {
    width: 120,
    height: 3,
    borderRadius: 2,
    backgroundColor: "#d4d0cc",
  },
});