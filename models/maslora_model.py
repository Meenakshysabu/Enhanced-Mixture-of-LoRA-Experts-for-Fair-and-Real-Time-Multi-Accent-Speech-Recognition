import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, in_features)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling

class MultiHeadAttentionWithLoRA(nn.Module):
    """Multi-head attention with LoRA adaptation"""
    def __init__(self, hidden_dim, num_heads, lora_rank=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Base projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # LoRA adaptations
        self.q_lora = LoRALayer(hidden_dim, hidden_dim, lora_rank, lora_alpha, dropout)
        self.k_lora = LoRALayer(hidden_dim, hidden_dim, lora_rank, lora_alpha, dropout)
        self.v_lora = LoRALayer(hidden_dim, hidden_dim, lora_rank, lora_alpha, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Apply base projections + LoRA
        q = self.q_proj(x) + self.q_lora(x)
        k = self.k_proj(x) + self.k_lora(x)
        v = self.v_proj(x) + self.v_lora(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(out)

class FeedForwardWithLoRA(nn.Module):
    """Feed-forward network with LoRA adaptation"""
    def __init__(self, hidden_dim, ff_dim, lora_rank=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        
        # LoRA adaptations
        self.fc1_lora = LoRALayer(hidden_dim, ff_dim, lora_rank, lora_alpha, dropout)
        self.fc2_lora = LoRALayer(ff_dim, hidden_dim, lora_rank, lora_alpha, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x1 = F.relu(self.fc1(x) + self.fc1_lora(x))
        x1 = self.dropout(x1)
        x2 = self.fc2(x1) + self.fc2_lora(x1)
        return self.dropout(x2)

class TransformerBlockWithLoRA(nn.Module):
    """Transformer block with LoRA adaptation"""
    def __init__(self, hidden_dim, num_heads, ff_dim, lora_rank=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionWithLoRA(hidden_dim, num_heads, lora_rank, lora_alpha, dropout)
        self.feed_forward = FeedForwardWithLoRA(hidden_dim, ff_dim, lora_rank, lora_alpha, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x

class DurationPredictor(nn.Module):
    """Duration predictor for phoneme durations"""
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        durations = self.linear(x).squeeze(-1)  # (batch, seq_len)
        return F.softplus(durations)  # Ensure positive durations

class MASLoRAPlusPlus(nn.Module):
    """MAS-LoRA++ Model for TTS with Monotonic Alignment Search"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Phoneme embedding
        self.phoneme_embedding = nn.Embedding(
            config['phoneme']['num_phonemes'],
            config['phoneme']['phoneme_embedding_dim']
        )
        
        # Input projection
        self.input_proj = nn.Linear(
            config['phoneme']['phoneme_embedding_dim'],
            config['model']['hidden_dim']
        )
        
        # Transformer encoder with LoRA
        self.encoder_layers = nn.ModuleList([
            TransformerBlockWithLoRA(
                hidden_dim=config['model']['hidden_dim'],
                num_heads=config['model']['num_heads'],
                ff_dim=config['model']['hidden_dim'] * 4,
                lora_rank=config['model']['lora']['rank'],
                lora_alpha=config['model']['lora']['alpha'],
                dropout=config['model']['lora']['dropout']
            )
            for _ in range(config['model']['num_layers'])
        ])
        
        # Duration predictor
        self.duration_predictor = DurationPredictor(config['model']['hidden_dim'])
        
        # Mel-spectrogram decoder
        self.mel_decoder = nn.Sequential(
            nn.Linear(config['model']['hidden_dim'], config['model']['hidden_dim'] * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['model']['hidden_dim'] * 2, config['audio']['n_mels'])
        )
        
    def forward(self, phonemes, durations=None, inference=False):
        """
        Args:
            phonemes: (batch, seq_len) - phoneme indices
            durations: (batch, seq_len) - ground truth durations (for training)
            inference: bool - whether in inference mode
        """
        # Embed phonemes
        x = self.phoneme_embedding(phonemes)  # (batch, seq_len, embed_dim)
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Encode with transformer
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Predict durations
        pred_durations = self.duration_predictor(x)  # (batch, seq_len)
        
        # Use ground truth durations during training, predicted during inference
        if inference or durations is None:
            durations_to_use = pred_durations
        else:
            durations_to_use = durations
        
        # Length regulation (expand phoneme features based on durations)
        x_expanded = self.length_regulation(x, durations_to_use)
        
        # Decode to mel-spectrogram
        mel_output = self.mel_decoder(x_expanded)
        
        return {
            'mel': mel_output,
            'durations': pred_durations
        }
    
    def length_regulation(self, x, durations):
        """Expand phoneme features based on predicted durations"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Convert durations to frame counts
        duration_frames = torch.round(durations * 100).long()  # Assuming 100 frames per second
        duration_frames = torch.clamp(duration_frames, min=1)
        
        # Calculate output length
        max_len = duration_frames.sum(dim=1).max().item()
        
        # Expand features
        expanded = torch.zeros(batch_size, max_len, hidden_dim, device=x.device)
        
        for b in range(batch_size):
            pos = 0
            for i in range(seq_len):
                dur = duration_frames[b, i].item()
                if pos + dur > max_len:
                    dur = max_len - pos
                if dur > 0:
                    expanded[b, pos:pos+dur] = x[b, i].unsqueeze(0).repeat(dur, 1)
                    pos += dur
                if pos >= max_len:
                    break
        
        return expanded
    
    def freeze_base_model(self):
        """Freeze all parameters except LoRA parameters"""
        for name, param in self.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
