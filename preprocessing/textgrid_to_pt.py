import os
import torch
import torchaudio
import textgrid
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# === CONFIG ===
data_dir = Path("C:/Main/data/common_voice")
aligned_dir = data_dir / "aligned"
output_dir = Path("C:/Main/data/training_data")
output_dir.mkdir(exist_ok=True)

# Phoneme to index mapping (ARPA phoneme set)
PHONEME_SET = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T',
    'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SP'
]
phoneme_to_idx = {p: i for i, p in enumerate(PHONEME_SET)}
phoneme_to_idx[''] = len(PHONEME_SET)  # silence/unknown

def parse_textgrid(tg_path):
    """Parse TextGrid file and extract phoneme alignments"""
    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        
        # Find phones tier
        phones_tier = None
        for tier in tg.tiers:
            if 'phone' in tier.name.lower():
                phones_tier = tier
                break
        
        if phones_tier is None:
            return None
        
        # Extract phoneme sequence with timestamps
        phonemes = []
        durations = []
        
        for interval in phones_tier:
            phone = interval.mark.strip().upper()
            if phone and phone != '':
                # Map to standard phoneme set
                if phone in phoneme_to_idx:
                    phonemes.append(phoneme_to_idx[phone])
                else:
                    # Handle variants (e.g., AA0, AA1 -> AA)
                    base_phone = ''.join([c for c in phone if c.isalpha()])
                    if base_phone in phoneme_to_idx:
                        phonemes.append(phoneme_to_idx[base_phone])
                    else:
                        phonemes.append(phoneme_to_idx[''])
                
                duration = interval.maxTime - interval.minTime
                durations.append(duration)
        
        return {
            'phonemes': phonemes,
            'durations': durations
        }
    except Exception as e:
        print(f"Error parsing {tg_path}: {e}")
        return None

def process_sample(wav_path, tg_path, output_path):
    """Process a single audio-TextGrid pair into .pt format"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(str(wav_path))
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Parse TextGrid
        alignment = parse_textgrid(tg_path)
        if alignment is None:
            return False
        
        # Create training sample
        sample = {
            'audio': waveform,
            'sample_rate': sample_rate,
            'phonemes': torch.tensor(alignment['phonemes'], dtype=torch.long),
            'durations': torch.tensor(alignment['durations'], dtype=torch.float32),
            'filename': wav_path.stem
        }
        
        # Save as .pt
        torch.save(sample, output_path)
        return True
        
    except Exception as e:
        print(f"Error processing {wav_path.name}: {e}")
        return False

def process_directory(aligned_subdir, output_subdir):
    """Process all TextGrid files in a directory"""
    aligned_path = aligned_dir / aligned_subdir
    output_path = output_dir / output_subdir
    output_path.mkdir(exist_ok=True)
    
    if not aligned_path.exists():
        print(f"⚠️ Directory not found: {aligned_path}")
        return 0, 0
    
    # Find all TextGrid files
    tg_files = list(aligned_path.glob("*.TextGrid"))
    
    if len(tg_files) == 0:
        print(f"⚠️ No TextGrid files found in {aligned_path}")
        return 0, 0
    
    print(f"\n📂 Processing {len(tg_files)} files from {aligned_subdir}...")
    
    success = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        for tg_path in tg_files:
            # Find corresponding wav file
            wav_path = tg_path.with_suffix('.wav')
            if not wav_path.exists():
                # Try in original directories
                wav_name = tg_path.stem + '.wav'
                wav_path = data_dir / aligned_subdir / wav_name
            
            if wav_path.exists():
                pt_path = output_path / f"{tg_path.stem}.pt"
                futures.append(executor.submit(process_sample, wav_path, tg_path, pt_path))
            else:
                print(f"⚠️ Audio not found for {tg_path.name}")
                failed += 1
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            if future.result():
                success += 1
            else:
                failed += 1
    
    return success, failed

# === MAIN ===
print("🚀 Converting TextGrid + WAV to .pt format...\n")

total_success = 0
total_failed = 0

# Process aligned directories
for subdir in ['processed', 'augmented']:
    s, f = process_directory(subdir, subdir)
    total_success += s
    total_failed += f

print(f"\n✅ Conversion complete!")
print(f"✅ Successfully converted: {total_success} samples")
print(f"⚠️ Failed: {total_failed} samples")
print(f"📁 Training data saved to: {output_dir}")
