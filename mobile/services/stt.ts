//stt.ts
let sttMessageCallback: ((data: any) => void) | null = null;

let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];
let streamRef: MediaStream | null = null;
let isConnected = false;
let isRecording = false;

const getBackendHost = () => {
  return typeof window !== "undefined" ? window.location.hostname : "localhost";
};

export const connectSttSocket = (onMessage: (data: any) => void) => {
  sttMessageCallback = onMessage;
  isConnected = true;

  console.log("✅ STT web mode initialized");

  if (sttMessageCallback) {
    sttMessageCallback({
      type: "status",
      message: "STT web mode ready",
    });
  }
};

export const startSttListening = async () => {
  try {
    if (!isConnected) {
      console.log("⚠️ STT not initialized");
      return;
    }

    if (typeof window === "undefined" || !navigator.mediaDevices) {
      throw new Error("Microphone is not available in this environment.");
    }

    if (isRecording) {
      console.log("⚠️ Already recording");
      return;
    }

    recordedChunks = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef = stream;

    let mimeType = "audio/webm";
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = "";
    }

    mediaRecorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType } : undefined,
    );

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onerror = (event: any) => {
      console.log("❌ MediaRecorder error:", event);
      if (sttMessageCallback) {
        sttMessageCallback({
          type: "error",
          message: event?.error?.message || "Recording failed",
        });
      }
    };

    mediaRecorder.start();
    isRecording = true;

    console.log("🎤 Web recording started");

    if (sttMessageCallback) {
      sttMessageCallback({
        type: "level",
        level: 1,
        isRecording: true,
      });

      sttMessageCallback({
        type: "status",
        message: "Listening started",
      });
    }
  } catch (error: any) {
    console.log("❌ startSttListening error:", error);

    if (sttMessageCallback) {
      sttMessageCallback({
        type: "error",
        message: error?.message || "Failed to start microphone",
      });
    }
  }
};

export const stopSttListening = async () => {
  try {
    if (!mediaRecorder || !isRecording) {
      console.log("⚠️ No active recording");
      return;
    }

    console.log("🛑 Stopping web recording...");

    const audioBlob: Blob = await new Promise((resolve, reject) => {
      if (!mediaRecorder) {
        reject(new Error("No active recorder"));
        return;
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, {
          type: recordedChunks[0]?.type || "audio/webm",
        });
        resolve(blob);
      };

      mediaRecorder.onerror = (event: any) => {
        reject(new Error(event?.error?.message || "Recording failed"));
      };

      mediaRecorder.stop();
    });

    if (streamRef) {
      streamRef.getTracks().forEach((track) => track.stop());
      streamRef = null;
    }

    mediaRecorder = null;
    isRecording = false;

    if (sttMessageCallback) {
      sttMessageCallback({
        type: "level",
        level: 0,
        isRecording: false,
      });

      sttMessageCallback({
        type: "status",
        message: "Transcribing...",
      });
    }

    const formData = new FormData();
    formData.append("file", audioBlob, "speech.webm");

    const host = getBackendHost();
    const response = await fetch(`http://${host}:8000/stt`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`STT upload failed: ${response.status} ${errorText}`);
    }

    const result = await response.json();
    const text = (result?.text || "").trim();

    console.log("📝 STT result:", text);

    if (sttMessageCallback) {
      sttMessageCallback({
        type: "transcript",
        text: text || "…",
      });
    }
  } catch (error: any) {
    console.log("❌ stopSttListening error:", error);

    if (streamRef) {
      streamRef.getTracks().forEach((track) => track.stop());
      streamRef = null;
    }

    mediaRecorder = null;
    isRecording = false;
    recordedChunks = [];

    if (sttMessageCallback) {
      sttMessageCallback({
        type: "error",
        message: error?.message || "STT failed",
      });

      sttMessageCallback({
        type: "level",
        level: 0,
        isRecording: false,
      });
    }
  }
};

export const closeSttSocket = () => {
  console.log("🔌 STT web mode closed");

  if (streamRef) {
    streamRef.getTracks().forEach((track) => track.stop());
    streamRef = null;
  }

  mediaRecorder = null;
  recordedChunks = [];
  isConnected = false;
  isRecording = false;
  sttMessageCallback = null;
};