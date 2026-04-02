// MainScreen.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Animated,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { closeSocket, connectSocket } from "../services/socket";
import {
  closeSttSocket,
  connectSttSocket,
  startSttListening,
  stopSttListening,
} from "../services/stt";

import AudioWave from "../components/AudioWave";
import CameraComponent from "../components/CameraView";

// TTS plays through Pi's MAX 98375A speaker via backend speak()
const SENTENCE_DISPLAY_MS = 4000;

export default function MainScreen() {
  const [activeTab, setActiveTab] = useState<"sign" | "speech">("sign");

  // ── Sign tab state (dynamic) ──────────────────────────────────────────────
  const [signedWords, setSignedWords] = useState<string[]>([]);
  const [finalSentence, setFinalSentence] = useState("");
  const [signStatus, setSignStatus] = useState("Waiting for sign...");

  // ── Speech tab state ─────────────────────────────────────────────────────
  const [isRecording, setIsRecording] = useState(false);
  const [sttText, setSttText] = useState("Say something...");
  const [isSpeechListening, setIsSpeechListening] = useState(false);

  const autoClearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Animations ────────────────────────────────────────────────────────────
  const sentenceFade = useRef(new Animated.Value(0)).current;
  const sentenceSlide = useRef(new Animated.Value(16)).current;
  const chipsFade = useRef(new Animated.Value(0)).current;
  const sentenceOut = useRef(new Animated.Value(1)).current;

  const animateSentenceIn = () => {
    sentenceOut.setValue(1);
    sentenceFade.setValue(0);
    sentenceSlide.setValue(16);

    Animated.parallel([
      Animated.timing(sentenceFade, {
        toValue: 1,
        duration: 380,
        useNativeDriver: true,
      }),
      Animated.timing(sentenceSlide, {
        toValue: 0,
        duration: 380,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const animateSentenceOut = (onDone: () => void) => {
    Animated.timing(sentenceOut, {
      toValue: 0,
      duration: 500,
      useNativeDriver: true,
    }).start(onDone);
  };

  const animateChipIn = () => {
    chipsFade.setValue(0);
    Animated.timing(chipsFade, {
      toValue: 1,
      duration: 180,
      useNativeDriver: true,
    }).start();
  };

  const scheduleAutoClear = () => {
    if (autoClearTimerRef.current) clearTimeout(autoClearTimerRef.current);

    autoClearTimerRef.current = setTimeout(() => {
      animateSentenceOut(() => {
        setFinalSentence("");
        setSignedWords([]);
        setSignStatus("Waiting for sign...");
        sentenceFade.setValue(0);
        sentenceOut.setValue(1);
      });
    }, SENTENCE_DISPLAY_MS);
  };

  useEffect(() => {
    if (activeTab === "sign") {
      connectSocket((data) => {
        console.log("📩 WS message:", data);

        const isReady = data?.is_ready === true;
        const label = data?.top1_label ?? "";
        const conf = data?.top1_conf ?? 0;
        const dbg = data?.debug ?? {};

        const sentence = data?.sentence_english;
        if (typeof sentence === "string" && sentence.trim().length > 0) {
          setFinalSentence(sentence);
          setSignedWords([]);
          animateSentenceIn();
          setSignStatus("💬 Speaking on Pi speaker...");
          scheduleAutoClear();
          return;
        }

        const isCollecting = label === "Collecting...";
        const isWaiting = label === "Waiting...";
        const isTooShort = label === "Too short / ignored";

        if (isCollecting || dbg.collecting) {
          setSignStatus(`✋ Signing... (${dbg.frames_collected ?? 0} frames)`);
          return;
        }

        if (dbg.hands_detected && !isReady) {
          setSignStatus("👋 Hand detected...");
          return;
        }

        if (!isReady || isWaiting || isTooShort) {
          setSignStatus((prev) =>
            prev.startsWith("💬") ? prev : "Waiting for sign...",
          );
          return;
        }

        if (typeof label === "string" && label.length > 0 && conf >= 0.4) {
          setSignedWords((prev) => [...prev, label]);
          animateChipIn();
          setSignStatus(`✅ Recognized: ${label}`);
        }
      });
    } else {
      closeSocket();
    }

    return () => {
      closeSocket();
      if (autoClearTimerRef.current) clearTimeout(autoClearTimerRef.current);
    };
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "speech") {
      closeSttSocket();
      setIsRecording(false);
      setIsSpeechListening(false);
      return;
    }

    connectSttSocket((data) => {
      console.log("🎤 STT message:", data);

      if (data?.type === "level") {
        setIsRecording(!!data.isRecording);
      }

      if (data?.type === "transcript") {
        const text = (data.text || "").trim();
        setSttText(text || "…");
        setIsSpeechListening(false);
        setIsRecording(false);
      }

      if (data?.type === "error") {
        setSttText(data.message || "STT error.");
        setIsRecording(false);
        setIsSpeechListening(false);
      }
    });

    return () => {
      if (activeTab !== "speech") {
        closeSttSocket();
      }
    };
  }, [activeTab]);

  const handleSpeechToggle = () => {
    if (isSpeechListening) {
      stopSttListening();
      setIsRecording(false);
    } else {
      setSttText("Listening...");
      setIsSpeechListening(true);
      startSttListening();
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.mainWrapper}>
        <View style={styles.header}>
          <Text style={styles.brandText}>E C H I F Y</Text>
        </View>

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
                <View style={styles.statusBar}>
                  <View
                    style={[
                      styles.statusDot,
                      signStatus.startsWith("✅")
                        ? styles.dotGreen
                        : signStatus.startsWith("✋")
                          ? styles.dotYellow
                          : signStatus.startsWith("💬")
                            ? styles.dotBlue
                            : signStatus.startsWith("👋")
                              ? styles.dotYellow
                              : styles.dotGray,
                    ]}
                  />
                  <Text style={styles.statusText} numberOfLines={1}>
                    {signStatus}
                  </Text>
                </View>

                {signedWords.length > 0 && (
                  <Animated.View
                    style={[styles.signedSection, { opacity: chipsFade }]}
                  >
                    <Text style={styles.sectionLabel}>SIGNS DETECTED</Text>
                    <View style={styles.chipRow}>
                      {signedWords.map((w, i) => (
                        <View key={`${w}-${i}`} style={styles.chip}>
                          <Text style={styles.chipText}>{w}</Text>
                        </View>
                      ))}
                    </View>
                  </Animated.View>
                )}

                {signedWords.length > 0 && <View style={styles.divider} />}

                <View style={styles.sentenceBlock}>
                  <Text style={styles.translationLabel}>FSL TRANSLATION</Text>
                  <ScrollView showsVerticalScrollIndicator={false}>
                    {finalSentence ? (
                      <Animated.View
                        style={{
                          opacity: Animated.multiply(sentenceFade, sentenceOut),
                          transform: [{ translateY: sentenceSlide }],
                        }}
                      >
                        <Text style={styles.translationText}>
                          {finalSentence}
                        </Text>
                      </Animated.View>
                    ) : (
                      <Text style={styles.placeholderText}>
                        {signedWords.length > 0
                          ? "Keep signing..."
                          : "Waiting for sign..."}
                      </Text>
                    )}
                  </ScrollView>
                </View>
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
                      isSpeechListening
                        ? styles.stopButton
                        : styles.startButton,
                    ]}
                    onPress={handleSpeechToggle}
                  >
                    <Text style={styles.toggleButtonText}>
                      {isSpeechListening ? "Stop" : "Start"}
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>

              <View style={styles.speechBottomPanel}>
                <Text style={styles.translationLabel}>SPEECH TO TEXT</Text>
                <ScrollView showsVerticalScrollIndicator={false}>
                  <Text style={styles.translationText}>{sttText}</Text>
                </ScrollView>
              </View>
            </View>
          )}
        </View>
      </View>
    </View>
  );
}

const THEME = {
  primary: "#8B4E1D",
  background: "#F7F5F3",
  panel: "#FFFFFF",
  text: "#1D2A39",
  muted: "#7C7A76",
  border: "#E0DDD9",
  success: "#4CAF50",
  danger: "#C85A54",
  yellow: "#E6A817",
  blue: "#3B7DD8",
  panelAlt: "#EAE6E2",
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: THEME.background },
  mainWrapper: { flex: 1, paddingHorizontal: 20, paddingTop: 10, paddingBottom: 20 },
  header: { position: "absolute", left: 20, top: 18, zIndex: 10 },
  brandText: { fontSize: 18, fontWeight: "800", color: THEME.primary, letterSpacing: 1 },
  tabContainer: {
    flexDirection: "row",
    backgroundColor: "#E2E0DD",
    borderRadius: 34,
    padding: 4,
    marginBottom: 12,
    width: "60%",
    maxWidth: 300,
    alignSelf: "flex-end",
  },
  tab: { flex: 1, height: 30, borderRadius: 60, justifyContent: "center", alignItems: "center" },
  activeTab: { backgroundColor: THEME.primary },
  tabText: { fontSize: 11, fontWeight: "700", color: THEME.muted },
  activeTabText: { color: "#FFFFFF" },
  content: { flex: 1 },
  signLayout: { flex: 1, flexDirection: "row", gap: 12 },
  cameraPanel: {
    flex: 1.6,
    backgroundColor: "#000",
    borderRadius: 18,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: THEME.border,
  },
  translationPanel: {
    flex: 0.7,
    backgroundColor: THEME.border,
    borderRadius: 18,
    padding: 18,
    borderWidth: 1,
    borderColor: THEME.border,
    maxHeight: "100%",
  },
  statusBar: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#fff",
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 5,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: THEME.border,
    gap: 6,
  },
  statusDot: { width: 7, height: 7, borderRadius: 4 },
  dotGreen: { backgroundColor: THEME.success },
  dotYellow: { backgroundColor: THEME.yellow },
  dotBlue: { backgroundColor: THEME.blue },
  dotGray: { backgroundColor: "#ccc" },
  statusText: { fontSize: 10, fontWeight: "600", color: THEME.muted, flex: 1 },
  signedSection: { marginBottom: 4 },
  sectionLabel: { fontSize: 8, fontWeight: "800", color: THEME.muted, letterSpacing: 2, marginBottom: 6 },
  chipRow: { flexDirection: "row", flexWrap: "wrap", gap: 4, marginBottom: 5 },
  chip: {
    backgroundColor: THEME.primary + "15",
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderWidth: 1,
    borderColor: THEME.primary + "30",
  },
  chipText: { fontSize: 10, fontWeight: "700", color: THEME.primary },
  pauseHint: { fontSize: 9, color: "#bbb", fontStyle: "italic" },
  divider: { height: 1, backgroundColor: THEME.border, marginVertical: 10 },
  sentenceBlock: { flex: 1 },
  translationLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: THEME.primary,
    marginBottom: 6,
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  translationText: { fontSize: 16, fontWeight: "700", color: THEME.text, lineHeight: 26 },
  placeholderText: { fontSize: 13, fontWeight: "500", color: "#bbb", fontStyle: "italic", lineHeight: 20 },
  speechLayout: { flex: 1, gap: 12 },
  speechTopPanel: { height: 60, paddingHorizontal: 15, justifyContent: "center" },
  waveRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  waveWrapper: { flex: 1, height: 30, justifyContent: "center", alignItems: "center", overflow: "hidden" },
  toggleButton: { minWidth: 70, height: 30, borderRadius: 15, justifyContent: "center", alignItems: "center" },
  startButton: { backgroundColor: THEME.success },
  stopButton: { backgroundColor: THEME.danger },
  toggleButtonText: { color: "#FFFFFF", fontSize: 13, fontWeight: "800" },
  speechBottomPanel: {
    flex: 1,
    backgroundColor: THEME.border,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: THEME.border,
    padding: 24,
  },
});