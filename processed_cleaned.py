# === 📦 Install Dependencies ===
import os
import librosa
import noisereduce as nr
import soundfile as sf

# === 🔧 Input and Output Folders ===
input_dir = r"C:\Main\data\common_voice\clips"
output_dir = r"C:\Main\data\common_voice\processed"

# === 🛡️ Check if input folder exists ===
if not os.path.exists(input_dir):
    print(f"❌ Input folder not found: {input_dir}")
else:
    os.makedirs(output_dir, exist_ok=True)

    # === ✅ Clean Audio Function ===
    def clean_audio(audio_path, output_path=None, target_sr=16000):
        try:
            y, sr = librosa.load(audio_path, sr=None)
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)
            y_normalized = librosa.util.normalize(y_trimmed)
            y_resampled = librosa.resample(y_normalized, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y_normalized
            sr = target_sr
            if output_path:
                sf.write(output_path, y_resampled, sr)
            return True
        except Exception as e:
            print(f"❌ Error cleaning {audio_path}: {e}")
            return False

    # === 🚀 Smart Batch Clean Remaining MP3s ===
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".mp3", "_clean.wav"))

            if os.path.exists(output_path):
                print(f"⏭️ Skipped (already cleaned): {filename}")
                skipped_count += 1
                continue

            result = clean_audio(input_path, output_path)
            if result:
                print(f"✅ Cleaned: {filename}")
                success_count += 1
            else:
                fail_count += 1

    # === 📊 Final Summary ===
    print(f"\n✅ Cleaning complete.")
    print(f"   Total MP3 files: {success_count + fail_count + skipped_count}")
    print(f"   Cleaned: {success_count}")
    print(f"   Skipped: {skipped_count}")
    print(f"   Failed: {fail_count}")
