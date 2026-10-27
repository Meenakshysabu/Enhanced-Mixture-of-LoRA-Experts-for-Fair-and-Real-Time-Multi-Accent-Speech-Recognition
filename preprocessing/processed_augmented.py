import os
import numpy as np
import librosa
import soundfile as sf

# === ✅ MAS-LoRA++ Augmentation Function ===
def augment_audio(audio, sr, accent=None, save_dir=None, base_name="sample"):
    augmented = []

    # === 1. Accent-aware pitch shift ===
    accent_shift = {
        "indian": 2,
        "british": -1,
        "australian": 1,
        "us": 0
    }
    n_steps = accent_shift.get(accent, 0)
    aug_pitch = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
    augmented.append(("pitch", aug_pitch))

    # === 2. Time stretch ===
    aug_stretch = librosa.effects.time_stretch(audio, rate=1.1)
    augmented.append(("stretch", aug_stretch))

    # === 3. Add Gaussian noise ===
    aug_noise = audio + 0.005 * np.random.randn(len(audio))
    augmented.append(("noise", aug_noise))

    # === 4. Phoneme dropout simulation ===
    dropout_ratio = 0.1
    mask = np.random.binomial(1, 1 - dropout_ratio, len(audio))
    aug_dropout = audio * mask
    augmented.append(("dropout", aug_dropout))

    # === 5. Contrastive pair (pitch + noise) ===
    aug_contrast = librosa.effects.pitch_shift(y=aug_noise, sr=sr, n_steps=-2)
    augmented.append(("contrast", aug_contrast))

    # === Save augmented files ===
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, (label, aug) in enumerate(augmented):
            out_path = os.path.join(save_dir, f"{base_name}_aug{i+1}.wav")
            sf.write(out_path, aug, sr)
            print(f"✅ Saved: {os.path.basename(out_path)} | Type: {label}")

    return augmented

# === 🚀 Batch Augment All Cleaned Files ===
input_dir = r"C:\Main\data\common_voice\processed"
output_dir = r"C:\Main\data\common_voice\augmented"
accent = "indian"

file_count = 0
aug_count = 0

for filename in os.listdir(input_dir):
    if filename.endswith("_clean.wav"):
        path = os.path.join(input_dir, filename)
        base_name = filename.replace("_clean.wav", "")
        try:
            audio, sr = librosa.load(path, sr=16000)
            augmented = augment_audio(audio, sr, accent=accent, save_dir=output_dir, base_name=base_name)
            file_count += 1
            aug_count += len(augmented)
        except Exception as e:
            print(f"❌ Failed to augment {filename}: {e}")

print(f"\n✅ Augmentation complete.")
print(f"   Files processed: {file_count}")
print(f"   Augmented samples saved: {aug_count}")



