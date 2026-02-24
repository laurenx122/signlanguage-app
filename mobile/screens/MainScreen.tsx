// mainscreen.tsx
import { Audio } from "expo-av";
import * as Speech from "expo-speech";
import React, { useEffect, useRef, useState } from "react";
import { Alert, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { connectSocket, closeSocket } from "../services/socket";
import { sendAudioForSTT } from "../services/stt";

import AudioWave from "../components/AudioWave";
import CameraComponent from "../components/CameraView";

export default function MainScreen() {
  const [activeTab, setActiveTab] = useState<"sign" | "speech">("sign");

  // ✅ text we show while spelling
const [typedText, setTypedText] = useState("");
const [finalWord, setFinalWord] = useState(""); // last spoken word
const newWordStartedRef = useRef(false);
  // Speech-to-text
  const [isRecording, setIsRecording] = useState(false);
  const [sttText, setSttText] = useState("Say something...");

  const recordingRef = useRef<Audio.Recording | null>(null);
  const silenceTimerRef = useRef(0);
  const speechStartedRef = useRef(false);
  const speechDurationRef = useRef(0);

  const isStoppingRef = useRef(false);
  const isUploadingRef = useRef(false);

  // keep a ref so websocket callback always sees latest text
  const typedRef = useRef<string>("");
  useEffect(() => {
    typedRef.current = typedText;
  }, [typedText]);

  // ✅ Connect sign websocket once
useEffect(() => {
  if (activeTab === "sign") {
    // ✅ DO NOT clear signText when entering sign tab
    // You may clear typedText if you want fresh spelling each time:
    // setTypedText("");

    connectSocket(async (data) => {
        const committed = data?.committed_letter;

        // ✅ When the first letter of a new word is committed:
        if (typeof committed === "string" && committed.length > 0) {
          if (!newWordStartedRef.current) {
            // first committed letter after a speak -> start new word
            setFinalWord("");       // ✅ remove previous word from view
            setTypedText("");       // reset buffer
            newWordStartedRef.current = true;
          }

          setTypedText((prev) => (prev || "") + committed);
          return;
        }

        // Optional: sync queue_text (but never overwrite with empty)
        const q = data?.queue_text;
        if (typeof q === "string" && q.length > 0) {
          // If new word started, keep showing the live letters
          setTypedText(q);
        }

        // ✅ Speak event (end of word)
        if (data?.should_speak && Array.isArray(data?.letters_to_speak)) {
          const word = data.letters_to_speak.join("");

          if (word.length > 0) {
            try {
              await Speech.stop();
              await Speech.speak(word, { language: "en-US", rate: 0.9, pitch: 1.0 });
              console.log("🔊 Speaking word:", word);
            } catch (e) {
              console.log("❌ Speech error:", e);
            }

            // ✅ show the finished word (persist until next spelling starts)
            setFinalWord(word);
          }

          // clear live buffer
          setTypedText("");

          // ✅ allow next session to clear finalWord on first new letter
          newWordStartedRef.current = false;
        }
      });
  } else {
    closeSocket();
  }

  return () => closeSocket();
}, [activeTab]);

  // --- Start/Stop STT when switching tabs ---
  useEffect(() => {
    if (activeTab === "speech") {
      startSpeechLoop();
    } else {
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

      rec.setProgressUpdateInterval(100);

      const silenceDbThreshold = -35;
      const silenceSecondsToStop = 1.0;
      const minSpeechSeconds = 0.3;
      const frameSec = 0.1;

      rec.setOnRecordingStatusUpdate((status) => {
        if (!status.isRecording) return;

        const db = (status as any).metering;
        if (typeof db !== "number") return;

        if (db > silenceDbThreshold) {
          speechStartedRef.current = true;
          silenceTimerRef.current = 0;
          speechDurationRef.current += frameSec;
        } else {
          if (speechStartedRef.current) silenceTimerRef.current += frameSec;
        }

        if (
          speechStartedRef.current &&
          speechDurationRef.current >= minSpeechSeconds &&
          silenceTimerRef.current >= silenceSecondsToStop
        ) {
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

      if (!restartAfter) return;

      if (isUploadingRef.current) return;
      isUploadingRef.current = true;

      setSttText((prev) =>
        prev && prev !== "Say something..." ? prev : "Transcribing..."
      );

      const text = await sendAudioForSTT(uri);

      setSttText((prev) => {
        const t = (text || "").trim();
        if (!t) return prev || "…";
        if (!prev || prev === "Say something..." || prev === "Listening..." || prev === "Transcribing...")
          return t;
        return t ? t : "…";
      });
    } catch (e) {
      console.log("❌ stop/send error:", e);
      setSttText((prev) => (prev ? prev : "STT error."));
    } finally {
      isUploadingRef.current = false;
      isStoppingRef.current = false;

      if (restartAfter && activeTab === "speech") {
        await startRecording();
      }
    }
  };

const signBoxText =
  typedText.length > 0
    ? typedText
    : finalWord.length > 0
    ? finalWord
    : "Waiting for sign...";

  return (
    <View style={styles.container}>
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === "sign" && styles.activeTab]}
          onPress={() => setActiveTab("sign")}
        >
          <Text style={[styles.tabText, activeTab === "sign" && styles.activeTabText]}>
            Sign to Speech
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tab, activeTab === "speech" && styles.activeTab]}
          onPress={() => setActiveTab("speech")}
        >
          <Text style={[styles.tabText, activeTab === "speech" && styles.activeTabText]}>
            Speech to Text
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.content}>
        {activeTab === "sign" ? (
          <View style={styles.stackedContainer}>
            <View style={styles.cameraBox}>
              <CameraComponent />
            </View>

            <View style={styles.fullTextBox}>
              <Text style={styles.predictionText}>{signBoxText}</Text>
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
    backgroundColor: "#fff" 
  },
  tabContainer: { 
    flexDirection: "row", 
    backgroundColor: "#e5e0db", 
    borderRadius: 25, 
    padding: 5 
  },
  tab: { 
    flex: 1, 
    paddingVertical: 12, 
    alignItems: "center", 
    borderRadius: 20 
  },
  activeTab: { 
    backgroundColor: "#6d3d1e" 
  },
  tabText: { 
    color: "#777", 
    fontWeight: "600" 
  },
  activeTabText: { 
    color: "#fff" 
  },
  content: { 
    marginTop: 20, 
    flex: 1 
  },
  stackedContainer: { 
    flexDirection: "column", 
    gap: 15, 
    height: "100%" 
  },
  cameraBox: { 
    height: 300, 
    borderRadius: 20, 
    overflow: "hidden" 
  },
  fullTextBox: { 
    backgroundColor: "#e5e0db", 
    height: 200, 
    borderRadius: 20, 
    padding: 20, 
    justifyContent: "center", 
    alignItems: "center" 
  },
  predictionText: { 
    fontSize: 28, 
    fontWeight: "700", 
    textAlign: "center", 
    color: "#333" 
  },
});