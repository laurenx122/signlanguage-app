//socket.ts

let socket: WebSocket | null = null;
let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
let messageCallback: ((data: any) => void) | null = null;
let manuallyClosed = false;

const getWsUrl = () => {
  const host =
    typeof window !== "undefined" ? window.location.hostname : "localhost";
  return `ws://${host}:8000/ws/fsl-dynamic`;
};

export const connectSocket = (onMessage: (data: any) => void) => {
  messageCallback = onMessage;
  manuallyClosed = false;

  if (
    socket &&
    (socket.readyState === WebSocket.OPEN ||
      socket.readyState === WebSocket.CONNECTING)
  ) {
    return;
  }

  const WS_URL = getWsUrl();

  console.log("🌐 Connecting to WebSocket:", WS_URL);
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    console.log("✅ WebSocket connected:", WS_URL);
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
  };

  socket.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (messageCallback) messageCallback(data);
    } catch {
      if (messageCallback) messageCallback({ prediction: String(e.data) });
    }
  };

  socket.onerror = () => {
    console.log("❌ WebSocket error");
  };

  socket.onclose = (event) => {
    console.log(
      "🔌 WebSocket closed",
      JSON.stringify({
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean,
        manuallyClosed,
      }),
    );

    socket = null;

    if (!manuallyClosed && !reconnectTimeout && messageCallback) {
      reconnectTimeout = setTimeout(() => {
        reconnectTimeout = null;
        if (messageCallback) connectSocket(messageCallback);
      }, 3000);
    }
  };
};

let sentCount = 0;

export const sendFrame = (frameBase64: string) => {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    if (sentCount < 5) {
      console.log("⚠️ sendFrame skipped: socket not open");
    }
    return false;
  }

  try {
    socket.send(frameBase64);
    sentCount += 1;

    if (sentCount <= 5 || sentCount % 30 === 0) {
      console.log(`📤 Sent frame #${sentCount}, length=${frameBase64.length}`);
    }

    return true;
  } catch (e) {
    console.log("❌ sendFrame failed", e);
    return false;
  }
};

export const closeSocket = () => {
  manuallyClosed = true;

  if (reconnectTimeout) clearTimeout(reconnectTimeout);
  reconnectTimeout = null;
  messageCallback = null;

  socket?.close();
  socket = null;
};