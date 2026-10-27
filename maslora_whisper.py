"""
MAS-LoRA++ with Whisper Integration
Includes: Phoneme modeling, Contrastive learning, and GeoLexicon for accent-aware ASR
"""

import os
# Force offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig
import math

class LoRALayer(nn.Module):
    """Enhanced LoRA layer with dropout and scaling"""
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices with proper initialization
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling

class PhonemeEncoder(nn.Module):
    """Phoneme-aware encoder for accent modeling"""
    def __init__(self, num_phonemes=42, phoneme_dim=128, hidden_dim=256):
        super().__init__()
        self.phoneme_embedding = nn.Embedding(num_phonemes, phoneme_dim)
        self.phoneme_proj = nn.Linear(phoneme_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, phoneme_ids):
        """
        Args:
            phoneme_ids: (batch, seq_len) - phoneme indices
        Returns:
            phoneme_features: (batch, seq_len, hidden_dim)
        """
        phoneme_emb = self.phoneme_embedding(phoneme_ids)
        phoneme_feat = self.phoneme_proj(phoneme_emb)
        return self.layer_norm(phoneme_feat)

class ContrastiveLearningHead(nn.Module):
    """Contrastive learning head for accent discrimination"""
    def __init__(self, hidden_dim=256, projection_dim=128, temperature=0.07):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.temperature = temperature
        
    def forward(self, features):
        """
        Args:
            features: (batch, seq_len, hidden_dim)
        Returns:
            projections: (batch, projection_dim)
        """
        # Global average pooling
        pooled = features.mean(dim=1)  # (batch, hidden_dim)
        projections = self.projection(pooled)  # (batch, projection_dim)
        # L2 normalize
        projections = F.normalize(projections, dim=-1)
        return projections
    
    def compute_contrastive_loss(self, z1, z2, labels=None):
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        Args:
            z1, z2: (batch, projection_dim) - two views of the same samples
            labels: (batch,) - accent labels for supervised contrastive learning
        """
        batch_size = z1.shape[0]
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # (2*batch, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2*batch, 2*batch)
        
        # Create mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Positive pairs: (i, i+batch) and (i+batch, i)
        pos_indices = torch.arange(batch_size, device=z.device)
        pos_sim = torch.cat([
            sim_matrix[pos_indices, pos_indices + batch_size],
            sim_matrix[pos_indices + batch_size, pos_indices]
        ])
        
        # Compute loss
        all_sim = torch.cat([
            sim_matrix[:batch_size],
            sim_matrix[batch_size:]
        ], dim=0)
        
        loss = -pos_sim + torch.logsumexp(all_sim, dim=1)
        return loss.mean()

class GeoLexicon(nn.Module):
    """Geographic Lexicon for accent-aware pronunciation modeling"""
    def __init__(self, num_accents=10, num_phonemes=42, hidden_dim=256):
        super().__init__()
        self.num_accents = num_accents
        self.num_phonemes = num_phonemes
        
        # Accent embedding
        self.accent_embedding = nn.Embedding(num_accents, hidden_dim)
        
        # Phoneme-accent interaction matrix
        self.phoneme_accent_matrix = nn.Parameter(
            torch.randn(num_accents, num_phonemes, num_phonemes) * 0.01
        )
        
        # Accent-specific pronunciation bias
        self.accent_bias = nn.Parameter(torch.zeros(num_accents, num_phonemes))
        
    def forward(self, phoneme_features, accent_ids):
        """
        Apply accent-specific transformations to phoneme features
        Args:
            phoneme_features: (batch, seq_len, hidden_dim)
            accent_ids: (batch,) - accent indices
        Returns:
            adapted_features: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = phoneme_features.shape
        
        # Get accent embeddings
        accent_emb = self.accent_embedding(accent_ids)  # (batch, hidden_dim)
        
        # Broadcast accent embedding to sequence
        accent_emb = accent_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_dim)
        
        # Combine phoneme features with accent information
        adapted_features = phoneme_features + accent_emb
        
        return adapted_features
    
    def get_accent_phoneme_mapping(self, accent_id, phoneme_id):
        """Get accent-specific phoneme transformation"""
        return self.phoneme_accent_matrix[accent_id, phoneme_id]

class MASLoRAPlusPlusWhisper(nn.Module):
    """
    MAS-LoRA++ with Whisper for Accent-Aware ASR
    Features:
    - Phoneme-level modeling
    - Contrastive learning for accent discrimination
    - GeoLexicon for accent-aware pronunciation
    - LoRA for efficient fine-tuning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained Whisper model
        whisper_config = WhisperConfig.from_pretrained(config['whisper']['model_name'])
        self.whisper = WhisperModel.from_pretrained(config['whisper']['model_name'])
        
        # Freeze Whisper encoder (we'll only train LoRA adapters)
        for param in self.whisper.encoder.parameters():
            param.requires_grad = False
        
        hidden_dim = whisper_config.d_model
        
        # Phoneme encoder
        self.phoneme_encoder = PhonemeEncoder(
            num_phonemes=config['phoneme']['num_phonemes'],
            phoneme_dim=config['phoneme']['phoneme_dim'],
            hidden_dim=hidden_dim
        )
        
        # GeoLexicon for accent-aware modeling
        self.geolexicon = GeoLexicon(
            num_accents=config['accent']['num_accents'],
            num_phonemes=config['phoneme']['num_phonemes'],
            hidden_dim=hidden_dim
        )
        
        # Contrastive learning head
        self.contrastive_head = ContrastiveLearningHead(
            hidden_dim=hidden_dim,
            projection_dim=config['contrastive']['projection_dim'],
            temperature=config['contrastive']['temperature']
        )
        
        # LoRA adapters for Whisper encoder layers
        self.lora_adapters = nn.ModuleList()
        for layer_idx in range(len(self.whisper.encoder.layers)):
            # Add LoRA to attention and FFN
            lora_attn_q = LoRALayer(
                hidden_dim, hidden_dim,
                rank=config['lora']['rank'],
                alpha=config['lora']['alpha'],
                dropout=config['lora']['dropout']
            )
            lora_attn_v = LoRALayer(
                hidden_dim, hidden_dim,
                rank=config['lora']['rank'],
                alpha=config['lora']['alpha'],
                dropout=config['lora']['dropout']
            )
            lora_ffn = LoRALayer(
                hidden_dim, hidden_dim * 4,
                rank=config['lora']['rank'],
                alpha=config['lora']['alpha'],
                dropout=config['lora']['dropout']
            )
            self.lora_adapters.append(nn.ModuleDict({
                'attn_q': lora_attn_q,
                'attn_v': lora_attn_v,
                'ffn': lora_ffn
            }))
        
        # Phoneme-aware decoder head
        self.phoneme_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, config['phoneme']['num_phonemes'])
        )
        
        # Accent classifier (auxiliary task)
        self.accent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, config['accent']['num_accents'])
        )
        
    def forward(self, input_features, phoneme_ids=None, accent_ids=None, 
                decoder_input_ids=None, return_contrastive=False):
        """
        Forward pass
        Args:
            input_features: (batch, n_mels, time) - mel-spectrogram
            phoneme_ids: (batch, seq_len) - phoneme sequence
            accent_ids: (batch,) - accent labels
            decoder_input_ids: (batch, seq_len) - decoder input for Whisper
            return_contrastive: bool - whether to return contrastive features
        """
        batch_size = input_features.shape[0]
        
        # Encode audio with Whisper encoder
        encoder_outputs = self.whisper.encoder(input_features)
        audio_features = encoder_outputs.last_hidden_state  # (batch, time, hidden_dim)
        
        # Apply LoRA adapters
        adapted_features = audio_features
        for layer_idx, lora_layer in enumerate(self.lora_adapters):
            # This is a simplified version - in practice, you'd integrate LoRA into attention
            # For now, we add LoRA residuals
            lora_residual = lora_layer['attn_q'](adapted_features)
            adapted_features = adapted_features + lora_residual
        
        # Phoneme encoding (if provided)
        if phoneme_ids is not None:
            phoneme_features = self.phoneme_encoder(phoneme_ids)
            
            # Apply GeoLexicon accent adaptation
            if accent_ids is not None:
                phoneme_features = self.geolexicon(phoneme_features, accent_ids)
            
            # Fuse audio and phoneme features
            # Align lengths using interpolation
            if adapted_features.shape[1] != phoneme_features.shape[1]:
                phoneme_features = F.interpolate(
                    phoneme_features.transpose(1, 2),
                    size=adapted_features.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            # Fusion
            fused_features = adapted_features + 0.3 * phoneme_features
        else:
            fused_features = adapted_features
        
        # Decode with Whisper decoder
        if decoder_input_ids is not None:
            decoder_outputs = self.whisper.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=fused_features
            )
            # Project to vocabulary space using the projection layer
            hidden_states = decoder_outputs.last_hidden_state
            logits = self.whisper.proj_out(hidden_states) if hasattr(self.whisper, 'proj_out') else hidden_states @ self.whisper.decoder.embed_tokens.weight.T
        else:
            logits = None
        
        # Phoneme prediction
        phoneme_logits = self.phoneme_decoder(fused_features)
        
        # Accent classification
        accent_logits = self.accent_classifier(fused_features.mean(dim=1))
        
        # Contrastive features
        contrastive_features = None
        if return_contrastive:
            contrastive_features = self.contrastive_head(fused_features)
        
        return {
            'logits': logits,
            'phoneme_logits': phoneme_logits,
            'accent_logits': accent_logits,
            'encoder_features': fused_features,
            'contrastive_features': contrastive_features
        }
    
    def freeze_whisper(self):
        """Freeze Whisper parameters (train only LoRA)"""
        for param in self.whisper.parameters():
            param.requires_grad = False
    
    def unfreeze_whisper(self):
        """Unfreeze Whisper parameters"""
        for param in self.whisper.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
