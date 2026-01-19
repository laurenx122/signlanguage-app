import { useEffect, useRef } from "react";

const SERVER_IP = "192.168.1.5"; // 🔴 CHANGE to your laptop IP
const WS_URL = `ws://${SERVER_IP}:8000/ws/sign`;

export function useSignSocket(onSign: (sign: string) => void) {
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      console.log("✅ Connected to Python backend");
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.sign) {
        onSign(data.sign);
      }
    };

    socket.onerror = (e) => {
      console.log("❌ WebSocket error", e);
    };

    return () => socket.close();
  }, []);
}
