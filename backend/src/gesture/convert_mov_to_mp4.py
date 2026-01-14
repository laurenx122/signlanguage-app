import os
import subprocess

# Full path to your ffmpeg.exe
FFMPEG_PATH = r"C:\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

ROOT = "data/raw/fsl105/clips"

for root, _, files in os.walk(ROOT):
    for file in files:
        if file.lower().endswith(".mov"):
            src = os.path.join(root, file)
            dst = os.path.splitext(src)[0] + ".mp4"

            if os.path.exists(dst):
                continue

            print("Converting:", src)

            try:
                subprocess.run(
                    [
                        FFMPEG_PATH, "-y",
                        "-i", src,
                        "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p",
                        dst
                    ],
                    check=True
                )
                print("✅ Converted:", dst)
            except subprocess.CalledProcessError as e:
                print("❌ Failed to convert:", src)
                print(e)
