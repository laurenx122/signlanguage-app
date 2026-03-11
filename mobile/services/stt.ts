import { Platform } from "react-native";

const STT_URL = "http://localhost:8000/stt";

export async function sendAudioForSTT(uri: string): Promise<string> {
  console.log("🎤 Uploading audio:", uri);

  const formData = new FormData();

  if (Platform.OS === "web") {
    // On web, convert the recorded URI/blob URL into a real Blob
    const fileResponse = await fetch(uri);
    const blob = await fileResponse.blob();

    formData.append("file", blob, "audio.webm");
  } else {
    // On Android/iOS, keep the React Native file object style
    formData.append("file", {
      uri,
      name: "audio.m4a",
      type: "audio/mp4",
    } as any);
  }

  let res: Response;

  try {
    res = await fetch(STT_URL, {
      method: "POST",
      body: formData,
    });
  } catch (e) {
    console.log("❌ fetch failed:", e);
    throw e;
  }

  console.log("✅ STT status:", res.status);

  const textBody = await res.text();
  console.log("✅ STT raw body:", textBody);

  try {
    const json = JSON.parse(textBody);
    return json.text ?? "";
  } catch {
    return textBody;
  }
}