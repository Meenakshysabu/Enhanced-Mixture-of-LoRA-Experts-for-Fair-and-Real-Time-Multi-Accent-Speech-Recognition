import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
import random

class TTSDataset(Dataset):
    """Dataset for TTS training with phoneme alignments"""
    def __init__(self, data_dir, config, augment=False):
        self.data_dir = Path(data_dir)
        self.config = config
        self.augment = augment
        
        # Load all .pt files
        self.samples = list(self.data_dir.glob("*.pt"))
        
        # Filter by max length
        self.samples = self._filter_by_length()
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _filter_by_length(self):
        """Filter samples by maximum audio length"""
        max_length = self.config['data']['max_audio_length']
        filtered = []
        
        for sample_path in self.samples:
            try:
                sample = torch.load(sample_path)
                audio_length = sample['audio'].shape[1] / sample['sample_rate']
                if audio_length <= max_length:
                    filtered.append(sample_path)
            except Exception as e:
                print(f"Error loading {sample_path}: {e}")
        
        return filtered
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load sample
        sample = torch.load(self.samples[idx])
        
        # Extract data
        audio = sample['audio'].squeeze(0)  # (seq_len,)
        phonemes = sample['phonemes']  # (num_phonemes,)
        durations = sample['durations']  # (num_phonemes,)
        
        # Compute mel-spectrogram
        mel_spec = self._compute_mel_spectrogram(audio, sample['sample_rate'])
        
        # Data augmentation
        if self.augment:
            mel_spec = self._augment_mel(mel_spec)
        
        return {
            'phonemes': phonemes,
            'durations': durations,
            'mel': mel_spec,
            'audio': audio
        }
    
    def _compute_mel_spectrogram(self, audio, sample_rate):
        """Compute mel-spectrogram from audio"""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.config['audio']['n_fft'],
            hop_length=self.config['audio']['hop_length'],
            win_length=self.config['audio']['win_length'],
            n_mels=self.config['audio']['n_mels'],
            f_min=self.config['audio']['fmin'],
            f_max=self.config['audio']['fmax']
        )
        
        mel = mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))  # Log scale
        
        return mel.transpose(0, 1)  # (time, n_mels)
    
    def _augment_mel(self, mel):
        """Apply data augmentation to mel-spectrogram"""
        # Time masking
        if random.random() < 0.5:
            time_mask_size = random.randint(1, min(20, mel.shape[0] // 10))
            time_mask_start = random.randint(0, max(0, mel.shape[0] - time_mask_size))
            mel[time_mask_start:time_mask_start + time_mask_size, :] = 0
        
        # Frequency masking
        if random.random() < 0.5:
            freq_mask_size = random.randint(1, min(10, mel.shape[1] // 10))
            freq_mask_start = random.randint(0, max(0, mel.shape[1] - freq_mask_size))
            mel[:, freq_mask_start:freq_mask_start + freq_mask_size] = 0
        
        return mel

def collate_fn(batch):
    """Collate function for batching variable-length sequences"""
    # Sort by phoneme length (descending)
    batch = sorted(batch, key=lambda x: len(x['phonemes']), reverse=True)
    
    # Get max lengths
    max_phoneme_len = max([len(x['phonemes']) for x in batch])
    max_mel_len = max([x['mel'].shape[0] for x in batch])
    mel_dim = batch[0]['mel'].shape[1]
    
    # Initialize tensors
    batch_size = len(batch)
    phonemes_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    durations_padded = torch.zeros(batch_size, max_phoneme_len)
    mel_padded = torch.zeros(batch_size, max_mel_len, mel_dim)
    
    phoneme_lengths = []
    mel_lengths = []
    
    # Fill tensors
    for i, sample in enumerate(batch):
        phoneme_len = len(sample['phonemes'])
        mel_len = sample['mel'].shape[0]
        
        phonemes_padded[i, :phoneme_len] = sample['phonemes']
        durations_padded[i, :phoneme_len] = sample['durations']
        mel_padded[i, :mel_len, :] = sample['mel']
        
        phoneme_lengths.append(phoneme_len)
        mel_lengths.append(mel_len)
    
    return {
        'phonemes': phonemes_padded,
        'durations': durations_padded,
        'mel': mel_padded,
        'phoneme_lengths': torch.tensor(phoneme_lengths),
        'mel_lengths': torch.tensor(mel_lengths)
    }

def create_dataloaders(config):
    """Create train and validation dataloaders"""
    train_dataset = TTSDataset(
        config['data']['train_dir'],
        config,
        augment=True
    )
    
    val_dataset = TTSDataset(
        config['data']['val_dir'],
        config,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
