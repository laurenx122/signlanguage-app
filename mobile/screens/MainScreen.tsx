import { Audio } from "expo-av";
import * as Speech from "expo-speech";
import React, { useEffect, useRef, useState } from "react";
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { connectSocket, closeSocket } from "../services/socket";
import { sendAudioForSTT } from "../services/stt";

import AudioWave from "../components/AudioWave";
import CameraComponent from "../components/CameraView";

export default function MainScreen() {
  const [activeTab, setActiveTab] = useState<"sign" | "speech">("sign");

  const [typedText, setTypedText] = useState("");
  const [finalWord, setFinalWord] = useState("");
  const newWordStartedRef = useRef(false);

  const [isRecording, setIsRecording] = useState(false);
  const [sttText, setSttText] = useState("Say something...");
  const [hasMicPermission, setHasMicPermission] = useState<boolean | null>(null);

  const recordingRef = useRef<Audio.Recording | null>(null);
  const silenceTimerRef = useRef(0);
  const speechStartedRef = useRef(false);
  const speechDurationRef = useRef(0);

  const isStoppingRef = useRef(false);
  const isUploadingRef = useRef(false);
  const shouldContinueSpeechRef = useRef(false);

  const speakWord = (word: string) => {
    const text = word.trim();
    if (!text) return;

    Speech.stop();
    Speech.speak(text, {
      language: "en-US",
      pitch: 1.0,
      rate: 0.9,
    });
  };

  const requestMicPermission = async () => {
    try {
      const current = await Audio.getPermissionsAsync();

      if (current.granted) {
        setHasMicPermission(true);
        return true;
      }

      const requested = await Audio.requestPermissionsAsync();
      const granted = requested.granted;

      setHasMicPermission(granted);

      if (!granted) {
        setSttText("Microphone permission denied.");
        return false;
      }

      return true;
    } catch (e) {
      console.log("requestMicPermission error:", e);
      setHasMicPermission(false);
      setSttText("Mic permission error.");
      return false;
    }
  };

  useEffect(() => {
    if (activeTab === "sign") {
      connectSocket(async (data) => {
        const committed = data?.committed_letter;

        if (typeof committed === "string" && committed.length > 0) {
          if (!newWordStartedRef.current) {
            setFinalWord("");
            setTypedText("");
            newWordStartedRef.current = true;
          }

          setTypedText((prev) => (prev || "") + committed);
          return;
        }

        const q = data?.queue_text;
        if (typeof q === "string" && q.length > 0) {
          setTypedText(q);
        }

        if (data?.should_speak && Array.isArray(data?.letters_to_speak)) {
          const word = data.letters_to_speak.join("").trim();

          if (word.length > 0) {
            setFinalWord(word);
            speakWord(word);
          }

          setTypedText("");
          newWordStartedRef.current = false;
        }
      });
    } else {
      closeSocket();
      Speech.stop();
    }

    return () => {
      closeSocket();
      Speech.stop();
    };
  }, [activeTab]);

  useEffect(() => {
    const setupSpeechTab = async () => {
      if (activeTab === "speech") {
        shouldContinueSpeechRef.current = true;
        await requestMicPermission();
      } else {
        shouldContinueSpeechRef.current = false;
        await stopRecordingAndSend(false, false);
        setIsRecording(false);
      }
    };

    setupSpeechTab();

    return () => {
      shouldContinueSpeechRef.current = false;
      stopRecordingAndSend(false, false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  const startSpeechLoop = async () => {
    try {
      const granted = await requestMicPermission();
      if (!granted) return;

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      if (shouldContinueSpeechRef.current) {
        await startRecording();
      }
    } catch (e) {
      console.log("startSpeechLoop error:", e);
      setSttText("Mic permission error.");
    }
  };

  const startRecording = async () => {
    try {
      if (!shouldContinueSpeechRef.current) return;
      if (recordingRef.current) return;
      if (!hasMicPermission) return;

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
        } else if (speechStartedRef.current) {
          silenceTimerRef.current += frameSec;
        }

        if (
          speechStartedRef.current &&
          speechDurationRef.current >= minSpeechSeconds &&
          silenceTimerRef.current >= silenceSecondsToStop
        ) {
          if (isStoppingRef.current || isUploadingRef.current) return;
          stopRecordingAndSend(true, true);
        }
      });

      await rec.startAsync();
    } catch (e) {
      console.log("startRecording error:", e);
      recordingRef.current = null;
      setIsRecording(false);
      setSttText("Mic error.");
    }
  };

  const stopRecordingAndSend = async (
    restartAfter: boolean,
    shouldTranscribe: boolean
  ) => {
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

      if (!shouldTranscribe) return;

      if (isUploadingRef.current) return;
      isUploadingRef.current = true;

      setSttText((prev) =>
        prev && prev !== "Say something..." ? prev : "Transcribing..."
      );

      const text = await sendAudioForSTT(uri);

      setSttText((prev) => {
        const t = (text || "").trim();
        if (!t) return prev || "…";
        if (
          !prev ||
          prev === "Say something..." ||
          prev === "Listening..." ||
          prev === "Transcribing..."
        ) {
          return t;
        }
        return t || "…";
      });
    } catch (e) {
      console.log("stop/send error:", e);
      setSttText((prev) => (prev ? prev : "STT error."));
    } finally {
      isUploadingRef.current = false;
      isStoppingRef.current = false;

      if (restartAfter && activeTab === "speech" && shouldContinueSpeechRef.current) {
        await startRecording();
      }
    }
  };

  const handleSpeechToggle = async () => {
    if (isRecording) {
      await stopRecordingAndSend(false, true);
    } else {
      shouldContinueSpeechRef.current = true;
      await startSpeechLoop();
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

      <View style={styles.content}>
        {activeTab === "sign" ? (
          <View style={styles.signLayout}>
            <View style={styles.cameraPanel}>
              <CameraComponent />
            </View>

            <View style={styles.translationPanel}>
              <Text style={styles.translationLabel}>FSL TRANSLATION:</Text>
              <Text style={styles.translationText}>{signBoxText}</Text>
            </View>
          </View>
        ) : (
          <View style={styles.speechLayout}>
            <View style={styles.speechTopPanel}>
              <View style={styles.waveRow}>
                <View style={styles.waveWrapper}>
                  <AudioWave isRecording={isRecording} />
                </View>

                <TouchableOpacity
                  style={[
                    styles.toggleButton,
                    isRecording ? styles.stopButton : styles.startButton,
                  ]}
                  onPress={handleSpeechToggle}
                >
                  <Text style={styles.toggleButtonText}>
                    {isRecording ? "Stop" : "Start"}
                  </Text>
                </TouchableOpacity>
              </View>
            </View>

            <View style={styles.speechBottomPanel}>
              <Text style={styles.translationLabel}>SPEECH TO TEXT:</Text>
              <Text style={styles.translationText}>{sttText}</Text>
            </View>
          </View>
        )}
      </View>
    </View>
  );
}

const PRIMARY = "#8B4E1D";
const BG = "#F7F5F3";
const PANEL = "#EAE6E2";
const TEXT = "#1D2A39";
const MUTED = "#7C7A76";
const TAB_BG = "#D9D4CF";
const BORDER = "#D8D0C8";
const GREEN = "#4CAF50";
const RED = "#C85A54";

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BG,
    paddingHorizontal: 18,
    paddingTop: 10,
    paddingBottom: 18,
  },

  brand: {
    textAlign: "center",
    fontSize: 18,
    fontWeight: "800",
    letterSpacing: 7,
    color: PRIMARY,
    marginBottom: 10,
  },

  tabContainer: {
    flexDirection: "row",
    backgroundColor: TAB_BG,
    borderRadius: 34,
    padding: 5,
    marginBottom: 18,
  },

  tab: {
    flex: 1,
    height: 60,
    borderRadius: 30,
    justifyContent: "center",
    alignItems: "center",
  },

  activeTab: {
    backgroundColor: PRIMARY,
  },

  tabText: {
    fontSize: 14,
    fontWeight: "700",
    color: MUTED,
  },

  activeTabText: {
    color: "#FFFFFF",
  },

  content: {
    flex: 1,
  },

  signLayout: {
    flex: 1,
    flexDirection: "row",
    gap: 18,
    alignItems: "stretch",
  },

  cameraPanel: {
    flex: 1.75,
    backgroundColor: "#000",
    borderRadius: 28,
    overflow: "hidden",
    minHeight: 420,
  },

  translationPanel: {
    flex: 1,
    backgroundColor: PANEL,
    borderRadius: 28,
    borderWidth: 1,
    borderColor: BORDER,
    paddingHorizontal: 26,
    paddingVertical: 24,
    justifyContent: "flex-start",
    minHeight: 420,
  },

  speechLayout: {
    flex: 0,
  },

  speechTopPanel: {
    height: 160,
    backgroundColor: PANEL,
    borderRadius: 28,
    borderWidth: 1,
    borderColor: BORDER,
    justifyContent: "center",
    paddingHorizontal: 20,
  },

  waveRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 14,
  },

  waveWrapper: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },

  toggleButton: {
    minWidth: 92,
    height: 46,
    borderRadius: 23,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 18,
  },

  startButton: {
    backgroundColor: GREEN,
  },

  stopButton: {
    backgroundColor: RED,
  },

  toggleButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "800",
  },

  speechBottomPanel: {
    height: 340,
    backgroundColor: PANEL,
    borderRadius: 28,
    borderWidth: 1,
    borderColor: BORDER,
    paddingHorizontal: 28,
    paddingVertical: 20,
    justifyContent: "flex-start",
    marginTop: 18,
  },

  translationLabel: {
    fontSize: 14,
    fontWeight: "800",
    color: PRIMARY,
    marginBottom: 16,
  },

  translationText: {
    fontSize: 34,
    fontWeight: "800",
    color: TEXT,
    lineHeight: 42,
  },
});