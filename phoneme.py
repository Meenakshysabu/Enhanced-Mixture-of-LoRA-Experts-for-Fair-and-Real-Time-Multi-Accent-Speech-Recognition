from phonemizer import phonemize
import os

input_dir = "C:/Main/data/common_voice/processed"
for file in os.listdir(input_dir):
    if file.endswith(".txt") and not file.endswith("_phonemes.txt"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            lines = f.readlines()
            sentence = lines[0].strip()
            accent = lines[1].strip().split(":")[1].strip()
            phonemes = phonemize(sentence, language='en-us', backend='espeak')
        with open(os.path.join(input_dir, file.replace(".txt", "_phonemes.txt")), "w", encoding="utf-8") as f:
            f.write(f"{phonemes}\nAccent: {accent}")

