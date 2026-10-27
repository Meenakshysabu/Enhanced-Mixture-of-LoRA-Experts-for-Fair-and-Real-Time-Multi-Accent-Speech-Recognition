"""
Enhanced dataset loader for MAS-LoRA++ with Whisper
Includes phoneme alignments, accent labels, and contrastive pairs
"""

import os
# Force offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
import random
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperTokenizer

class AccentAwareASRDataset(Dataset):
    """Dataset for accent-aware ASR with phoneme alignments"""
    def __init__(self, data_dir, config, split='train', augment=False):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.augment = augment
        
        # Initialize Whisper feature extractor and tokenizer
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config['whisper']['model_name']
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(
            config['whisper']['model_name']
        )
        
        # Load all .pt files
        self.samples = list(self.data_dir.glob("*.pt"))
        
        # Filter by max length
        self.samples = self._filter_by_length()
        
        # Create accent mapping
        self.accent_to_idx = self._create_accent_mapping()
        
        # Phoneme set
        self.phoneme_set = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
            'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T',
            'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SP', 'UNK'
        ]
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        print(f"Found {len(self.accent_to_idx)} unique accents")
    
    def _filter_by_length(self):
        """Filter samples by maximum audio length"""
        max_length = self.config['data']['max_audio_length']
        filtered = []
        
        for sample_path in self.samples:
            try:
                sample = torch.load(sample_path)
                if 'audio' in sample:
                    audio_length = sample['audio'].shape[-1] / sample['sample_rate']
                    if audio_length <= max_length:
                        filtered.append(sample_path)
            except Exception as e:
                print(f"Error loading {sample_path}: {e}")
        
        return filtered
    
    def _create_accent_mapping(self):
        """Create mapping from accent names to indices"""
        accents = set()
        for sample_path in self.samples[:1000]:  # Sample subset for speed
            try:
                # Extract accent from filename (e.g., en_us_0001.pt -> en_us)
                filename = sample_path.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    accent = '_'.join(parts[:2])  # e.g., "en_us", "en_gb"
                    accents.add(accent)
            except:
                pass
        
        accent_list = sorted(list(accents))
        if len(accent_list) == 0:
            accent_list = ['unknown']
        
        return {accent: idx for idx, accent in enumerate(accent_list)}
    
    def _extract_accent_from_filename(self, filename):
        """Extract accent label from filename"""
        parts = filename.split('_')
        if len(parts) >= 2:
            accent = '_'.join(parts[:2])
            return self.accent_to_idx.get(accent, 0)
        return 0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load sample
        sample_path = self.samples[idx]
        sample = torch.load(sample_path)
        
        # Extract data
        audio = sample['audio'].squeeze() if 'audio' in sample else None
        phonemes = sample['phonemes'] if 'phonemes' in sample else torch.zeros(10, dtype=torch.long)
        durations = sample['durations'] if 'durations' in sample else torch.zeros(10)
        
        # Get accent label
        accent_id = self._extract_accent_from_filename(sample_path.stem)
        
        # Get transcript if available
        transcript = sample.get('transcript', '')
        
        # Process audio with Whisper feature extractor
        if audio is not None:
            # Ensure audio is the right sample rate
            sample_rate = sample.get('sample_rate', 16000)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio = resampler(audio)
            
            # Extract features
            input_features = self.feature_extractor(
                audio.numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.squeeze(0)
            
            # Data augmentation
            if self.augment:
                input_features = self._augment_features(input_features)
        else:
            # Create dummy features if audio not available
            input_features = torch.zeros(80, 3000)
        
        # Tokenize transcript
        if transcript:
            labels = self.tokenizer(
                transcript,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=448
            ).input_ids.squeeze(0)
        else:
            labels = torch.zeros(10, dtype=torch.long)
        
        return {
            'input_features': input_features,
            'phonemes': phonemes,
            'durations': durations,
            'accent_id': torch.tensor(accent_id, dtype=torch.long),
            'labels': labels,
            'filename': sample_path.stem
        }
    
    def _augment_features(self, features):
        """Apply data augmentation to input features"""
        # Time masking
        if random.random() < 0.5:
            time_mask_size = random.randint(1, min(20, features.shape[1] // 10))
            time_mask_start = random.randint(0, max(0, features.shape[1] - time_mask_size))
            features[:, time_mask_start:time_mask_start + time_mask_size] = 0
        
        # Frequency masking
        if random.random() < 0.5:
            freq_mask_size = random.randint(1, min(10, features.shape[0] // 10))
            freq_mask_start = random.randint(0, max(0, features.shape[0] - freq_mask_size))
            features[freq_mask_start:freq_mask_start + freq_mask_size, :] = 0
        
        return features

class ContrastiveASRDataset(AccentAwareASRDataset):
    """Dataset that returns contrastive pairs for contrastive learning"""
    def __init__(self, data_dir, config, split='train', augment=True):
        super().__init__(data_dir, config, split, augment)
        
        # Group samples by accent for contrastive sampling
        self.accent_groups = {}
        for idx, sample_path in enumerate(self.samples):
            accent_id = self._extract_accent_from_filename(sample_path.stem)
            if accent_id not in self.accent_groups:
                self.accent_groups[accent_id] = []
            self.accent_groups[accent_id].append(idx)
    
    def __getitem__(self, idx):
        # Get first sample
        sample1 = super().__getitem__(idx)
        
        # Get contrastive pair (same accent, different audio)
        accent_id = sample1['accent_id'].item()
        if accent_id in self.accent_groups and len(self.accent_groups[accent_id]) > 1:
            # Sample from same accent
            candidates = [i for i in self.accent_groups[accent_id] if i != idx]
            if candidates:
                idx2 = random.choice(candidates)
                sample2 = super().__getitem__(idx2)
            else:
                sample2 = sample1
        else:
            sample2 = sample1
        
        return {
            'sample1': sample1,
            'sample2': sample2
        }

def collate_fn_whisper(batch):
    """Collate function for Whisper-based batching with attention masks"""
    # Extract max lengths
    max_feature_len = max([x['input_features'].shape[1] for x in batch])
    max_phoneme_len = max([len(x['phonemes']) for x in batch])
    max_label_len = max([len(x['labels']) for x in batch])
    
    batch_size = len(batch)
    feature_dim = batch[0]['input_features'].shape[0]
    
    # Initialize tensors
    input_features = torch.zeros(batch_size, feature_dim, max_feature_len)
    phonemes = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    durations = torch.zeros(batch_size, max_phoneme_len)
    accent_ids = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.full((batch_size, max_label_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_feature_len, dtype=torch.long)
    
    feature_lengths = []
    phoneme_lengths = []
    label_lengths = []
    filenames = []
    
    # Fill tensors
    for i, sample in enumerate(batch):
        feat_len = sample['input_features'].shape[1]
        phone_len = len(sample['phonemes'])
        label_len = len(sample['labels'])
        
        input_features[i, :, :feat_len] = sample['input_features']
        phonemes[i, :phone_len] = sample['phonemes']
        durations[i, :phone_len] = sample['durations']
        accent_ids[i] = sample['accent_id']
        labels[i, :label_len] = sample['labels']
        attention_mask[i, :feat_len] = 1  # Set attention mask for valid tokens
        
        feature_lengths.append(feat_len)
        phoneme_lengths.append(phone_len)
        label_lengths.append(label_len)
        filenames.append(sample['filename'])
    
    return {
        'input_features': input_features,
        'phonemes': phonemes,
        'durations': durations,
        'accent_ids': accent_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'feature_lengths': torch.tensor(feature_lengths),
        'phoneme_lengths': torch.tensor(phoneme_lengths),
        'label_lengths': torch.tensor(label_lengths),
        'filenames': filenames
    }

def collate_fn_contrastive(batch):
    """Collate function for contrastive learning"""
    # Separate sample1 and sample2
    samples1 = [x['sample1'] for x in batch]
    samples2 = [x['sample2'] for x in batch]
    
    # Collate each separately
    batch1 = collate_fn_whisper(samples1)
    batch2 = collate_fn_whisper(samples2)
    
    return {
        'sample1': batch1,
        'sample2': batch2
    }

def create_dataloaders_whisper(config, use_contrastive=True):
    """Create train and validation dataloaders for Whisper-based training"""
    if use_contrastive:
        train_dataset = ContrastiveASRDataset(
            config['data']['train_dir'],
            config,
            split='train',
            augment=True
        )
        collate_fn = collate_fn_contrastive
    else:
        train_dataset = AccentAwareASRDataset(
            config['data']['train_dir'],
            config,
            split='train',
            augment=True
        )
        collate_fn = collate_fn_whisper
    
    val_dataset = AccentAwareASRDataset(
        config['data']['val_dir'],
        config,
        split='val',
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
        collate_fn=collate_fn_whisper,
        pin_memory=True
    )
    
    return train_loader, val_loader
