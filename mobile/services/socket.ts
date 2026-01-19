import * as Speech from 'expo-speech';

let socket: WebSocket | null = null;
let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
let messageCallback: ((msg: string) => void) | null = null;
// 10.81.255.64 change this with your IP4 address from your lappy
const WS_URL = "ws://10.81.255.64:8000/ws/fsl-simple";
let lastPrediction: string | null = null;

export const handleSpeech = async (prediction: string) => {
  // Ignore invalid predictions
  if (!prediction || prediction === 'UNKNOWN') {
    lastPrediction = null; // reset when hand is lost
    return;
  }

  // 🔑 Speak ONLY when prediction changes
  if (prediction === lastPrediction) return;

  lastPrediction = prediction;

  console.log('🔊 Speaking (frontend):', prediction);

  try {
    await Speech.stop(); // 🔑 prevents overlap & silence bugs
    await Speech.speak(`${prediction}`, {
      language: 'en-US',
      pitch: 1.0,
      rate: 0.9,
    });
  } catch (e) {
    console.log('❌ Speech error:', e);
  }
};
/**
 * CONNECT
 */
export const connectSocket = (onMessage: (msg: string) => void) => {
  messageCallback = onMessage;

  if (socket && socket.readyState === WebSocket.OPEN) return;

  console.log("🌐 Connecting to WebSocket:", WS_URL);
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    console.log("✅ WebSocket connected");
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
  };

  socket.onmessage = (e) => {
    try {
      // 1. Try to parse as JSON first (if backend sends the full object)
      const data = JSON.parse(e.data);
      const prediction = data.prediction || 'UNKNOWN';
      
      // Update UI with just the letter/UNKNOWN
      if (messageCallback) messageCallback(prediction);
      
      // Handle Speech using the flag from backend
      handleSpeech(prediction);

    } catch (error) {
      // 2. Fallback: If backend sends RAW TEXT (e.g. just "A" or "UNKNOWN")
      const rawPrediction = e.data;
      console.log('📩 Raw Prediction:', rawPrediction);
      
      if (messageCallback) messageCallback(rawPrediction);
      
      // Since it's raw text, we assume we should speak if it's a valid letter
      handleSpeech(rawPrediction);
    }
  };

  socket.onerror = (e) => console.log("❌ WebSocket error");

  socket.onclose = () => {
    console.log("🔌 WebSocket closed");
    socket = null;
    if (!reconnectTimeout) {
      reconnectTimeout = setTimeout(() => {
        reconnectTimeout = null;
        if (messageCallback) connectSocket(messageCallback);
      }, 3000);
    }
  };
};

export const sendFrame = (frameBase64: string) => {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  try {
    socket.send(frameBase64);
    return true;
  } catch { return false; }
};

export const closeSocket = () => {
  if (reconnectTimeout) clearTimeout(reconnectTimeout);
  reconnectTimeout = null;
  messageCallback = null;
  socket?.close();
  socket = null;
};