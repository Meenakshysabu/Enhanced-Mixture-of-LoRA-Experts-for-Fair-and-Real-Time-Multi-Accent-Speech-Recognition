import os
import pandas as pd
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
data_dir = "C:/Main/data/common_voice"
clips_dir = os.path.join(data_dir, "clips")
tsv_file = os.path.join(data_dir, "audio_verified.tsv")
output_dir = os.path.join(data_dir, "processed")
os.makedirs(output_dir, exist_ok=True)

# === LOAD METADATA ===
print("🔍 Loading metadata...")
df = pd.read_csv(tsv_file, sep="\t")

# Use 'locale' as proxy for accent
df["accent"] = df["locale"].fillna("unknown") if "locale" in df.columns else "unknown"
df = df.dropna(subset=["path", "sentence"])

# === CLEAN AND MATCH FILENAMES ===
actual_files = set(os.listdir(clips_dir))
df["path_clean"] = df["path"].astype(str).str.strip()
df = df[df["path_clean"].isin(actual_files)]
df["full_path"] = df["path_clean"].apply(lambda p: os.path.normpath(os.path.join(clips_dir, p)))

print(f"✅ Found {len(df)} matching audio samples.")

# === PROCESS AUDIO + TEXT IN PARALLEL ===
print("⚡ Resuming parallel conversion (5 sec clips)...")

def process_row(i, row):
    mp3_path = row["full_path"]
    accent = row["accent"].lower().replace("-", "_").replace(" ", "_")
    wav_path = os.path.join(output_dir, f"{accent}_{i}.wav")
    txt_path = os.path.join(output_dir, f"{accent}_{i}.txt")

    # Skip if already processed
    if os.path.exists(wav_path) and os.path.exists(txt_path):
        return True

    try:
        sentence = str(row.get("sentence", "")).strip()
        if not sentence or sentence.lower() in ["dummy script", "unknown", ""]:
            print(f"⚠️ Skipping empty or dummy transcript at index {i}")
            return False

        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio[:5000]  # Trim to first 5 seconds
        audio.export(wav_path, format="wav")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{sentence}\nAccent: {accent}")

        return True
    except Exception as e:
        print(f"⚠️ Error processing {mp3_path}: {e}")
        return False

success_count = 0
error_count = 0

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_row, i, row) for i, row in df.iterrows()]
    for future in as_completed(futures):
        if future.result():
            success_count += 1
        else:
            error_count += 1

print(f"\n✅ Preprocessing complete.")
print(f"🎧 Newly converted: {success_count} files")
print(f"⚠️ Skipped or errored: {error_count} files")
print("📁 Files saved to:", output_dir)
