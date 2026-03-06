const STT_URL = "http://10.191.173.64:8000/stt";

export async function sendAudioForSTT(uri: string): Promise<string> {
  console.log("🎤 Uploading audio:", uri);

  const formData = new FormData();
  formData.append("file", {
    uri,
    name: "audio.m4a",
    type: "audio/mp4",
  } as any);

  let res: Response;

  try {
    res = await fetch(STT_URL, {
      method: "POST",
      body: formData,
      // ✅ do NOT set Content-Type
    });
  } catch (e) {
    console.log("❌ fetch failed:", e);
    throw e;
  }

  console.log("✅ STT status:", res.status);

  const textBody = await res.text();        // ✅ read as text first
  console.log("✅ STT raw body:", textBody);

  try {
    const json = JSON.parse(textBody);      // ✅ parse manually
    return json.text ?? "";
  } catch {
    // If backend returns plain text, still support it:
    return textBody;
  }
}
