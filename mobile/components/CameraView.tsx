//CameraView.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Platform,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { sendFrame } from "../services/socket";

type CameraDevice = {
  deviceId: string;
  label: string;
};

interface CameraViewProps {
  onPrediction?: (prediction: string) => void;
}

export default function CameraView({ onPrediction }: CameraViewProps) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [errorText, setErrorText] = useState("");
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (Platform.OS !== "web") {
      setErrorText("This camera selector version is for web only.");
      setIsInitialized(true);
      return;
    }

    let mounted = true;

    const setup = async () => {
      try {
        // Ask permission first so labels become visible
        const tempStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        tempStream.getTracks().forEach((t) => t.stop());

        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = allDevices
          .filter((d) => d.kind === "videoinput")
          .map((d) => ({
            deviceId: d.deviceId,
            label: d.label || "Unnamed camera",
          }));

        if (!mounted) return;

        setDevices(videoInputs);

        const preferred =
          videoInputs.find((d) =>
            d.label.toLowerCase().includes("a4tech"),
          ) || videoInputs[0];

        if (preferred) {
          setSelectedDeviceId(preferred.deviceId);
        }

        setIsInitialized(true);
      } catch (err: any) {
        console.log("Camera init error:", err);
        if (!mounted) return;
        setErrorText(err?.message || "Failed to initialize camera.");
        setIsInitialized(true);
      }
    };

    setup();

    return () => {
      mounted = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (Platform.OS !== "web") return;
    if (!isInitialized || !selectedDeviceId) return;

    let active = true;

    const startCamera = async () => {
      try {
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: selectedDeviceId },
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });

        if (!active) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        setIsCapturing(true);
        setErrorText("");
      } catch (err: any) {
        console.log("Start camera error:", err);
        setErrorText(err?.message || "Failed to start selected camera.");
        setIsCapturing(false);
      }
    };

    startCamera();

    return () => {
      active = false;
    };
  }, [isInitialized, selectedDeviceId]);

  useEffect(() => {
    if (Platform.OS !== "web") return;
    if (!isCapturing) return;

    let active = true;

    const captureLoop = async () => {
      await new Promise((r) => setTimeout(r, 300));

      while (active) {
        try {
          const video = videoRef.current;
          const canvas = canvasRef.current;

          if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
            await new Promise((r) => setTimeout(r, 100));
            continue;
          }

          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          const ctx = canvas.getContext("2d");
          if (!ctx) {
            await new Promise((r) => setTimeout(r, 100));
            continue;
          }

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const dataUrl = canvas.toDataURL("image/jpeg", 0.45);
          const base64 = dataUrl.split(",")[1];

          if (base64) {
            sendFrame(base64);
          }
        } catch (err) {
          console.log("Frame capture error:", err);
        }

        await new Promise((r) => setTimeout(r, 120));
      }
    };

    captureLoop();

    return () => {
      active = false;
    };
  }, [isCapturing]);

  if (!isInitialized) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.loadingText}>Initializing camera...</Text>
      </View>
    );
  }

  if (errorText) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.loadingText}>Camera error</Text>
        <Text style={styles.loadingSubtext}>{errorText}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {Platform.OS === "web" && (
        <>
          <View style={styles.selectWrap}>
            <Text style={styles.selectLabel}>Camera:</Text>
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              style={selectStyle}
            >
              {devices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label}
                </option>
              ))}
            </select>
          </View>

          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={styles.video as any}
          />

          <canvas ref={canvasRef} style={{ display: "none" }} />

          <View style={styles.statusIndicator}>
            <View
              style={[styles.statusDot, isCapturing && styles.statusDotActive]}
            />
            <Text style={styles.statusText}>
              {isCapturing ? "Capturing..." : "Idle"}
            </Text>
          </View>
        </>
      )}
    </View>
  );
}

const selectStyle: React.CSSProperties = {
  height: 36,
  minWidth: 260,
  borderRadius: 8,
  border: "1px solid #ccc",
  padding: "0 10px",
  backgroundColor: "#fff",
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  video: {
    flex: 1,
    width: "100%",
    height: "100%",
    objectFit: "cover",
    backgroundColor: "#000",
  },
  selectWrap: {
    position: "absolute",
    top: 10,
    left: 10,
    zIndex: 20,
    backgroundColor: "rgba(255,255,255,0.92)",
    borderRadius: 10,
    padding: 8,
    gap: 6,
  },
  selectLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: "#333",
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f5f5f5",
    padding: 20,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 18,
    color: "#333",
    fontWeight: "600",
    textAlign: "center",
  },
  loadingSubtext: {
    marginTop: 8,
    fontSize: 14,
    color: "#666",
    textAlign: "center",
  },
  statusIndicator: {
    position: "absolute",
    top: 10,
    right: 10,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#ff4444",
    marginRight: 6,
  },
  statusDotActive: {
    backgroundColor: "#44ff44",
  },
  statusText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "600",
  },
});