import os
import pandas as pd

clips_dir = "C:/Main/data/common_voice/clips"
output_tsv = "C:/Main/data/common_voice/audio_verified.tsv"

# List all MP3 files
files = [f for f in os.listdir(clips_dir) if f.endswith(".mp3")]

# Create dummy metadata
df = pd.DataFrame({
    "path": files,
    "sentence": ["dummy transcript"] * len(files),
    "locale": ["en-IN"] * len(files)
})

df.to_csv(output_tsv, sep="\t", index=False)
print(f"✅ Created metadata for {len(df)} audio files.")

