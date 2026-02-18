import { Audio } from "expo-av";
import React, { useEffect, useRef, useState } from "react";
import { Alert, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { connectSocket, closeSocket } from "../services/socket";
import { sendAudioForSTT } from "../services/stt";

import AudioWave from "../components/AudioWave";
import CameraComponent from "../components/CameraView";

export default function MainScreen() {
  const [activeTab, setActiveTab] = useState<"sign" | "speech">("sign");

  // Sign-to-speech
  const [prediction, setPrediction] = useState("Waiting for sign...");

  // Speech-to-text
  const [isRecording, setIsRecording] = useState(false);
  const [sttText, setSttText] = useState("Say something...");

  // Recording refs
  const recordingRef = useRef<Audio.Recording | null>(null);
  const silenceTimerRef = useRef(0);
  const speechStartedRef = useRef(false);
  const speechDurationRef = useRef(0);

  // Locks to prevent double stop/upload overlaps (Android fix)
  const isStoppingRef = useRef(false);
  const isUploadingRef = useRef(false);

  // --- Connect sign websocket once ---
 useEffect(() => {
  if (activeTab === "sign") {
    connectSocket((msg) => setPrediction(msg));
  } else {
    closeSocket();
  }

  return () => {
    closeSocket();
  };
}, [activeTab]);

  // --- Start/Stop STT when switching tabs ---
  useEffect(() => {
    if (activeTab === "speech") {
      startSpeechLoop();
    } else {
      // leaving speech tab: stop recording but don't upload
      stopRecordingAndSend(false);
      setIsRecording(false);
    }

    return () => {
      stopRecordingAndSend(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  const startSpeechLoop = async () => {
    const { granted } = await Audio.requestPermissionsAsync();
    if (!granted) {
      Alert.alert(
        "Permission Required",
        "Please enable microphone access in settings to use Speech to Text."
      );
      setActiveTab("sign");
      return;
    }

    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
    });

    await startRecording();
  };

  const startRecording = async () => {
    try {
      // don't overwrite existing transcription
      setSttText((prev) =>
        prev && prev !== "Say something..." && prev !== "Listening..."
          ? prev
          : "Listening..."
      );

      setIsRecording(true);

      speechStartedRef.current = false;
      silenceTimerRef.current = 0;
      speechDurationRef.current = 0;

      const rec = new Audio.Recording();
      recordingRef.current = rec;

      await rec.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);

      // ✅ Android: consistent metering updates
      rec.setProgressUpdateInterval(100); // ms (≈0.1s)

      // ✅ Android silence detection settings
      const silenceDbThreshold = -35;   // try -30 if room is noisy, -40 if too sensitive
      const silenceSecondsToStop = 1.0; // stop after 1 second of silence
      const minSpeechSeconds = 0.3;     // must speak a bit before auto-stop
      const frameSec = 0.1;             // matches 100ms update interval

      rec.setOnRecordingStatusUpdate((status) => {
        if (!status.isRecording) return;

        const db = (status as any).metering;
        if (typeof db !== "number") return;

        // console.log("🎚 dB:", db);

        if (db > silenceDbThreshold) {
          speechStartedRef.current = true;
          silenceTimerRef.current = 0;
          speechDurationRef.current += frameSec;
        } else {
          if (speechStartedRef.current) {
            silenceTimerRef.current += frameSec;
          }
        }

        if (
          speechStartedRef.current &&
          speechDurationRef.current >= minSpeechSeconds &&
          silenceTimerRef.current >= silenceSecondsToStop
        ) {
          // ✅ prevent multiple stop calls
          if (isStoppingRef.current || isUploadingRef.current) return;
          stopRecordingAndSend(true);
        }
      });

      await rec.startAsync();
    } catch (e) {
      console.log("❌ startRecording error:", e);
      setIsRecording(false);
      setSttText("Mic error.");
    }
  };

  const stopRecordingAndSend = async (restartAfter: boolean) => {
    // ✅ prevent double stop
    if (isStoppingRef.current) return;
    isStoppingRef.current = true;

    const rec = recordingRef.current;
    if (!rec) {
      isStoppingRef.current = false;
      return;
    }

    try {
      recordingRef.current = null;
      setIsRecording(false);

      await rec.stopAndUnloadAsync();
      const uri = rec.getURI();

      if (!uri) {
        setSttText((prev) => prev || "No audio captured.");
        return;
      }

      // If we're leaving the tab (restartAfter=false), don't upload
      if (!restartAfter) return;

      // ✅ prevent overlapping uploads
      if (isUploadingRef.current) {
        // release stop lock so future stops work
        return;
      }
      isUploadingRef.current = true;

      setSttText((prev) =>
        prev && prev !== "Say something..." ? prev : "Transcribing..."
      );

      const text = await sendAudioForSTT(uri);

      // Append transcription (keeps history)
      setSttText((prev) => {
        const t = (text || "").trim();
        if (!t) return prev || "…";
        if (
          !prev ||
          prev === "Say something..." ||
          prev === "Listening..." ||
          prev === "Transcribing..."
        )
          return t;
         return t ? t : "…";
      });
    } catch (e) {
      console.log("❌ stop/send error:", e);
      setSttText((prev) => (prev ? prev : "STT error."));
    } finally {
      isUploadingRef.current = false;
      isStoppingRef.current = false;

      // ✅ restart only after upload finishes and still on speech tab
      if (restartAfter && activeTab === "speech") {
        await startRecording();
      }
    }
  };

  return (
    <View style={styles.container}>
      {/* Toggle Buttons */}
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === "sign" && styles.activeTab]}
          onPress={() => setActiveTab("sign")}
        >
          <Text
            style={[
              styles.tabText,
              activeTab === "sign" && styles.activeTabText,
            ]}
          >
            Sign to Speech
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tab, activeTab === "speech" && styles.activeTab]}
          onPress={() => setActiveTab("speech")}
        >
          <Text
            style={[
              styles.tabText,
              activeTab === "speech" && styles.activeTabText,
            ]}
          >
            Speech to Text
          </Text>
        </TouchableOpacity>
      </View>

      {/* Main Content Area */}
      <View style={styles.content}>
        {activeTab === "sign" ? (
          <View style={styles.stackedContainer}>
            <View style={styles.cameraBox}>
              <CameraComponent />
            </View>

            <View style={styles.fullTextBox}>
              <Text style={styles.predictionText}>{prediction}</Text>
            </View>
          </View>
        ) : (
          <View>
            <AudioWave isRecording={isRecording} />
            <View style={styles.fullTextBox}>
              <Text style={styles.predictionText}>{sttText}</Text>
            </View>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 0,
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: "#fff",
  },
  tabContainer: {
    flexDirection: "row",
    backgroundColor: "#e5e0db",
    borderRadius: 25,
    padding: 5,
  },
  tab: { flex: 1, paddingVertical: 12, alignItems: "center", borderRadius: 20 },
  activeTab: { backgroundColor: "#6d3d1e" },
  tabText: { color: "#777", fontWeight: "600" },
  activeTabText: { color: "#fff" },
  content: { marginTop: 20, flex: 1 },
  stackedContainer: { flexDirection: "column", gap: 15, height: "100%" },
  cameraBox: { height: 300, borderRadius: 20, overflow: "hidden" },
  fullTextBox: {
    backgroundColor: "#e5e0db",
    height: 200,
    borderRadius: 20,
    padding: 20,
    justifyContent: "center",
    alignItems: "center",
  },
  predictionText: {
    fontSize: 28,
    fontWeight: "700",
    textAlign: "center",
    color: "#333",
  },
});
