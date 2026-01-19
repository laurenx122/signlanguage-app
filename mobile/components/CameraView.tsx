import React, { useEffect, useRef, useState } from "react";
import { StyleSheet, Text, View, ActivityIndicator, TouchableOpacity, Linking, Platform } from "react-native";
import { Camera, useCameraDevice, useCameraPermission } from "react-native-vision-camera";
import { sendFrame } from "../services/socket";

interface CameraViewProps {
  onPrediction?: (prediction: string) => void;
}

export default function CameraView({ onPrediction }: CameraViewProps) {
  const device = useCameraDevice("front");
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);

  const [isCapturing, setIsCapturing] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  // -----------------------
  // Camera permission - Single request only
  // -----------------------
  useEffect(() => {
    let isMounted = true;

    const initializeCamera = async () => {
      if (isInitialized) {
        console.log("⏭️ Already initialized, skipping");
        return;
      }

      console.log("🔍 Permission status:", hasPermission);

      // If we already have permission, just mark as initialized
      if (hasPermission === true) {
        console.log("✅ Permission already granted");
        if (isMounted) {
          setIsInitialized(true);
        }
        return;
      }

      // If permission was denied, don't keep requesting
      if (hasPermission === false) {
        console.log("❌ Permission denied");
        if (isMounted) {
          setIsInitialized(true);
        }
        return;
      }

      // Only request if undefined (not asked yet)
      if (hasPermission === undefined) {
        console.log("📷 Requesting camera permission for the first time...");
        try {
          const granted = await requestPermission();
          console.log("📷 Permission result:", granted);
        } catch (error) {
          console.error("❌ Error requesting permission:", error);
        } finally {
          if (isMounted) {
            setIsInitialized(true);
          }
        }
      }
    };

    // Small delay to ensure React Native is ready
    const timer = setTimeout(() => {
      initializeCamera();
    }, 300);

    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, []); // Empty dependency array - only run once!

  // -----------------------
  // Capture loop
  // -----------------------
  useEffect(() => {
    if (!hasPermission || !device || !isInitialized) {
      console.log("⏸️ Skipping capture:", { 
        hasPermission, 
        hasDevice: !!device,
        isInitialized 
      });
      return;
    }

    let isActive = true;
    setIsCapturing(true);
    console.log("▶️ Starting capture loop");

    const captureLoop = async () => {
      // Wait for camera to be ready
      await new Promise(r => setTimeout(r, 1000));

      while (isActive) {
        try {
          if (!cameraRef.current) {
            await new Promise(r => setTimeout(r, 200));
            continue;
          }

          const snapshot = await cameraRef.current.takeSnapshot({
            quality: 85,  // Increased from 0.1 to 0.5 for better detection
            skipMetadata: true,
          });

          if (snapshot?.path) {
            const response = await fetch(`file://${snapshot.path}`);
            const blob = await response.blob();

            const base64 = await new Promise<string>((resolve) => {
              const reader = new FileReader();
              reader.onloadend = () => {
                const result = reader.result as string;
                resolve(result.split(",")[1]);
              };
              reader.readAsDataURL(blob);
            });

            sendFrame(base64);
          }
        } catch (err) {
          console.log("📸 Frame capture error:", err);
        }

        await new Promise(r => setTimeout(r, 300)); // Reduced to ~3 FPS for better quality
      }
    };

    captureLoop();

    return () => {
      console.log("⏹️ Stopping capture loop");
      isActive = false;
      setIsCapturing(false);
    };
  }, [hasPermission, device, isInitialized]);

  // -----------------------
  // Handle opening settings
  // -----------------------
  const openSettings = () => {
    Linking.openSettings();
  };

  const retryPermission = async () => {
    console.log("🔄 Manually retrying permission request...");
    const granted = await requestPermission();
    console.log("🔄 Retry result:", granted);
  };

  // -----------------------
  // RENDER
  // -----------------------
  
  // Still initializing
  if (!isInitialized) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.loadingText}>Initializing camera...</Text>
        <Text style={styles.loadingSubtext}>
          {hasPermission === undefined ? "Checking permissions..." : "Loading..."}
        </Text>
      </View>
    );
  }

  // Permission denied
  if (hasPermission === false) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.errorIcon}>📷</Text>
        <Text style={styles.errorTitle}>Camera Access Required</Text>
        <Text style={styles.errorText}>
          This app needs camera permission to recognize sign language gestures.
        </Text>
        <Text style={styles.errorSubtext}>
          Please grant camera permission in your device settings.
        </Text>
        
        <TouchableOpacity style={styles.settingsButton} onPress={openSettings}>
          <Text style={styles.settingsButtonText}>Open Settings</Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.retryButton} onPress={retryPermission}>
          <Text style={styles.retryButtonText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // No device found
  if (!device) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.loadingText}>Looking for camera...</Text>
        <Text style={styles.errorText}>
          No front camera detected on this device.
        </Text>
      </View>
    );
  }

  // Camera ready
  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={styles.camera}
        device={device}
        isActive={true}
        photo={true}
      />

      <View style={styles.statusIndicator}>
        <View style={[styles.statusDot, isCapturing && styles.statusDotActive]} />
        <Text style={styles.statusText}>
          {isCapturing ? "Capturing..." : "Idle"}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  camera: {
    flex: 1,
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
  },
  loadingSubtext: {
    marginTop: 8,
    fontSize: 14,
    color: "#666",
  },
  errorIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  errorTitle: {
    fontSize: 22,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 12,
    textAlign: "center",
  },
  errorText: {
    fontSize: 15,
    color: "#666",
    textAlign: "center",
    marginTop: 8,
    paddingHorizontal: 20,
    lineHeight: 22,
  },
  errorSubtext: {
    fontSize: 13,
    color: "#999",
    textAlign: "center",
    marginTop: 8,
    paddingHorizontal: 20,
    lineHeight: 20,
    fontStyle: "italic",
  },
  settingsButton: {
    marginTop: 24,
    backgroundColor: "#4CAF50",
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 8,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  settingsButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "700",
  },
  retryButton: {
    marginTop: 12,
    paddingHorizontal: 32,
    paddingVertical: 14,
  },
  retryButtonText: {
    color: "#4CAF50",
    fontSize: 16,
    fontWeight: "600",
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