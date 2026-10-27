import pandas as pd
import os

# === Paths ===
tsv_dir = r"C:\Main\data\common_voice"
tsv_files = ["validated.tsv", "other.tsv"]
output_dir = r"C:\Main\data\common_voice\txt"
os.makedirs(output_dir, exist_ok=True)

# === Load and combine transcripts ===
df_list = []
for file in tsv_files:
    path = os.path.join(tsv_dir, file)
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t")
        df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)
transcript_map = {row["path"].replace(".mp3", ""): row["sentence"] for _, row in df_all.iterrows()}

# === Generate .txt files for cleaned and augmented audio ===
base_dir = r"C:\Main\data\common_voice\processed"
for filename in os.listdir(base_dir):
    if filename.endswith("_clean.wav"):
        base = filename.replace("_clean.wav", "")
        if base in transcript_map:
            # Cleaned
            with open(os.path.join(output_dir, f"{base}_clean.txt"), "w", encoding="utf-8") as f:
                f.write(transcript_map[base])
            # Augmented
            for i in range(1, 6):
                aug_name = f"{base}_aug{i}"
                with open(os.path.join(output_dir, f"{aug_name}.txt"), "w", encoding="utf-8") as f:
                    f.write(transcript_map[base])

print("✅ Transcript generation complete.")
