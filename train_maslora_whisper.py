"""
Training script for MAS-LoRA++ with Whisper
Includes phoneme modeling, contrastive learning, and GeoLexicon
"""

import os
# Force offline mode to use cached models
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from transformers import WhisperForConditionalGeneration

from models.maslora_whisper import MASLoRAPlusPlusWhisper
from models.dataset_whisper import create_dataloaders_whisper
import jiwer

class MASLoRAWhisperTrainer:
    """Trainer for MAS-LoRA++ with Whisper"""
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize model
        print("Initializing MAS-LoRA++ model...")
        self.model = MASLoRAPlusPlusWhisper(self.config).to(self.device)
        
        # Initialize tokenizer for WER calculation
        from transformers import WhisperTokenizer
        self.tokenizer = WhisperTokenizer.from_pretrained(self.config['whisper']['model_name'])
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = self.model.get_trainable_parameters()
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Freeze Whisper if specified
        if self.config['whisper']['freeze_encoder']:
            self.model.freeze_whisper()
            print("Whisper encoder frozen (LoRA-only training)")
        
        # Create dataloaders
        print("Loading datasets...")
        self.train_loader, self.val_loader = create_dataloaders_whisper(
            self.config,
            use_contrastive=self.config['contrastive']['enabled']
        )
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss functions
        self.asr_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.phoneme_loss = nn.CrossEntropyLoss()
        self.accent_loss = nn.CrossEntropyLoss()
        self.duration_loss = nn.MSELoss()
        
        # Setup mixed precision training
        self.use_amp = self.config['training']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup logging
        if self.config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_wer = float('inf')
        
        # Auto-resume from latest checkpoint or specified checkpoint
        resume_path = self.config['checkpoint'].get('resume_from', None)
        auto_resume = self.config['checkpoint'].get('auto_resume', False)
        
        if resume_path:
            # Resume from specified checkpoint
            self.load_checkpoint(resume_path)
        else:
            # Auto-resume from latest checkpoint if available
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                print(f"Found existing checkpoint: {latest_checkpoint.name}")
                if auto_resume:
                    print("Auto-resuming from latest checkpoint...")
                    self.load_checkpoint(latest_checkpoint)
                else:
                    response = input("Resume from this checkpoint? (y/n): ").strip().lower()
                    if response == 'y':
                        self.load_checkpoint(latest_checkpoint)
                    else:
                        print("Starting training from scratch...")
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the checkpoint directory"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            # Get the checkpoint with the highest epoch number
            return checkpoints[-1]
        return None
    
    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_config = self.config['training']
        
        # Separate parameters for different learning rates
        lora_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {'params': lora_params, 'lr': optimizer_config['learning_rate']},
            {'params': other_params, 'lr': optimizer_config['learning_rate'] * 0.1}
        ]
        
        return AdamW(
            param_groups,
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            weight_decay=optimizer_config['weight_decay']
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_config = self.config['training']
        
        if scheduler_config['scheduler'] == 'cosine_with_warmup':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=scheduler_config['warmup_steps']
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'] * len(self.train_loader),
                eta_min=scheduler_config['min_lr']
            )
            
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[scheduler_config['warmup_steps']]
            )
        else:
            return None
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_asr_loss = 0
        total_phoneme_loss = 0
        total_accent_loss = 0
        total_contrastive_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        
        for batch_idx, batch in enumerate(pbar):
            # Handle contrastive batch structure
            if self.config['contrastive']['enabled']:
                batch1 = batch['sample1']
                batch2 = batch['sample2']
                batches = [batch1, batch2]
            else:
                batches = [batch]
            
            batch_loss = 0
            
            for batch_data in batches:
                # Move to device
                input_features = batch_data['input_features'].to(self.device)
                phonemes = batch_data['phonemes'].to(self.device)
                accent_ids = batch_data['accent_ids'].to(self.device)
                labels = batch_data['labels'].to(self.device)
                
                # Prepare decoder input ids and labels (clamp to valid vocab range)
                # Use the actual embedding layer size, not config.vocab_size
                vocab_size = self.model.whisper.decoder.embed_tokens.num_embeddings
                pad_token_id = self.model.whisper.config.pad_token_id
                
                # Clamp labels to valid range and replace invalid tokens
                labels_clamped = labels.clone()
                # First, mark positions that should be ignored
                ignore_mask = (labels_clamped == -100) | (labels_clamped >= vocab_size) | (labels_clamped < 0)
                # Clamp all values to valid range
                labels_clamped = torch.clamp(labels_clamped, 0, vocab_size - 1)
                # Set ignore positions back to -100
                labels_clamped[ignore_mask] = -100
                
                decoder_input_ids = None
                if labels_clamped.shape[1] > 1:
                    decoder_input_ids = labels_clamped[:, :-1].clone()
                    # Replace -100 (ignore index) with pad token id for decoder input
                    decoder_input_ids[decoder_input_ids == -100] = pad_token_id
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_features=input_features,
                        phoneme_ids=phonemes,
                        accent_ids=accent_ids,
                        decoder_input_ids=decoder_input_ids,
                        return_contrastive=self.config['contrastive']['enabled']
                    )
                    
                    # Compute losses
                    loss = 0
                    
                    # ASR loss
                    if outputs['logits'] is not None:
                        logits = outputs['logits']
                        target_labels = labels_clamped[:, 1:].reshape(-1) if labels_clamped.shape[1] > 1 else labels_clamped.reshape(-1)
                        asr_loss = self.asr_loss(
                            logits.reshape(-1, logits.shape[-1]),
                            target_labels
                        )
                        loss += self.config['training']['loss_weights']['asr'] * asr_loss
                        total_asr_loss += asr_loss.item()
                    
                    # Phoneme loss
                    if self.config['phoneme']['use_phoneme_loss']:
                        phoneme_logits = outputs['phoneme_logits']
                        # Align phoneme predictions with targets
                        if phoneme_logits.shape[1] >= phonemes.shape[1]:
                            phoneme_pred = phoneme_logits[:, :phonemes.shape[1]]
                        else:
                            phoneme_pred = phoneme_logits
                            phonemes = phonemes[:, :phoneme_pred.shape[1]]
                        
                        phone_loss = self.phoneme_loss(
                            phoneme_pred.reshape(-1, phoneme_pred.shape[-1]),
                            phonemes.reshape(-1)
                        )
                        loss += self.config['training']['loss_weights']['phoneme'] * phone_loss
                        total_phoneme_loss += phone_loss.item()
                    
                    # Accent loss
                    if self.config['accent']['use_accent_loss']:
                        accent_logits = outputs['accent_logits']
                        acc_loss = self.accent_loss(accent_logits, accent_ids)
                        loss += self.config['training']['loss_weights']['accent'] * acc_loss
                        total_accent_loss += acc_loss.item()
                    
                    # Normalize by gradient accumulation steps
                    loss = loss / grad_accum_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                batch_loss += loss.item()
            
            # Contrastive loss (if enabled)
            if self.config['contrastive']['enabled'] and len(batches) == 2:
                with autocast(enabled=self.use_amp):
                    # Get contrastive features from both views
                    batch1 = batch['sample1']
                    batch2 = batch['sample2']
                    
                    input1 = batch1['input_features'].to(self.device)
                    input2 = batch2['input_features'].to(self.device)
                    phonemes1 = batch1['phonemes'].to(self.device)
                    phonemes2 = batch2['phonemes'].to(self.device)
                    accent1 = batch1['accent_ids'].to(self.device)
                    accent2 = batch2['accent_ids'].to(self.device)
                    
                    out1 = self.model(input1, phonemes1, accent1, return_contrastive=True)
                    out2 = self.model(input2, phonemes2, accent2, return_contrastive=True)
                    
                    z1 = out1['contrastive_features']
                    z2 = out2['contrastive_features']
                    
                    # Compute contrastive loss
                    if self.config['contrastive']['use_supervised']:
                        cont_loss = self.model.contrastive_head.compute_contrastive_loss(
                            z1, z2, labels=accent1
                        )
                    else:
                        cont_loss = self.model.contrastive_head.compute_contrastive_loss(z1, z2)
                    
                    cont_loss = cont_loss * self.config['training']['loss_weights']['contrastive']
                    cont_loss = cont_loss / grad_accum_steps
                
                if self.use_amp:
                    self.scaler.scale(cont_loss).backward()
                else:
                    cont_loss.backward()
                
                total_contrastive_loss += cont_loss.item()
                batch_loss += cont_loss.item()
            
            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
            
            total_loss += batch_loss
            
            # Logging
            if self.global_step % self.config['logging']['log_interval'] == 0:
                if self.writer:
                    self.writer.add_scalar('train/loss', batch_loss, self.global_step)
                    self.writer.add_scalar('train/asr_loss', total_asr_loss / (batch_idx + 1), self.global_step)
                    if self.config['phoneme']['use_phoneme_loss']:
                        self.writer.add_scalar('train/phoneme_loss', total_phoneme_loss / (batch_idx + 1), self.global_step)
                    if self.config['accent']['use_accent_loss']:
                        self.writer.add_scalar('train/accent_loss', total_accent_loss / (batch_idx + 1), self.global_step)
                    if self.config['contrastive']['enabled']:
                        self.writer.add_scalar('train/contrastive_loss', total_contrastive_loss / (batch_idx + 1), self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'asr': f"{total_asr_loss/(batch_idx+1):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                input_features = batch['input_features'].to(self.device)
                phonemes = batch['phonemes'].to(self.device)
                accent_ids = batch['accent_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Prepare decoder input ids and labels (same as training)
                vocab_size = self.model.whisper.decoder.embed_tokens.num_embeddings
                pad_token_id = self.model.whisper.config.pad_token_id
                
                labels_clamped = labels.clone()
                ignore_mask = (labels_clamped == -100) | (labels_clamped >= vocab_size) | (labels_clamped < 0)
                labels_clamped = torch.clamp(labels_clamped, 0, vocab_size - 1)
                labels_clamped[ignore_mask] = -100
                
                decoder_input_ids = None
                if labels_clamped.shape[1] > 1:
                    decoder_input_ids = labels_clamped[:, :-1].clone()
                    decoder_input_ids[decoder_input_ids == -100] = pad_token_id
                
                # Forward pass
                outputs = self.model(
                    input_features=input_features,
                    phoneme_ids=phonemes,
                    accent_ids=accent_ids,
                    decoder_input_ids=decoder_input_ids
                )
                
                # Compute loss
                if outputs['logits'] is not None:
                    logits = outputs['logits']
                    target_labels = labels_clamped[:, 1:].reshape(-1) if labels_clamped.shape[1] > 1 else labels_clamped.reshape(-1)
                    loss = self.asr_loss(
                        logits.reshape(-1, logits.shape[-1]),
                        target_labels
                    )
                    total_loss += loss.item()
                    
                    # Collect predictions for WER calculation
                    pred_ids = torch.argmax(logits, dim=-1)
                    
                    # Decode predictions and references to text
                    for i in range(pred_ids.shape[0]):
                        # Get prediction tokens (remove padding)
                        pred_tokens = pred_ids[i].cpu().tolist()
                        pred_tokens = [t for t in pred_tokens if t != pad_token_id and t != -100]
                        
                        # Get reference tokens (remove padding and ignore tokens)
                        ref_tokens = labels_clamped[i, 1:].cpu().tolist() if labels_clamped.shape[1] > 1 else labels_clamped[i].cpu().tolist()
                        ref_tokens = [t for t in ref_tokens if t != pad_token_id and t != -100]
                        
                        # Decode to text using Whisper tokenizer
                        try:
                            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                            ref_text = self.tokenizer.decode(ref_tokens, skip_special_tokens=True)
                        except:
                            # Fallback to token IDs if decoding fails
                            pred_text = " ".join([str(t) for t in pred_tokens[:50]])
                            ref_text = " ".join([str(t) for t in ref_tokens[:50]])
                        
                        predictions.append(pred_text)
                        references.append(ref_text)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute WER using jiwer
        wer = 0.0
        if len(predictions) > 0 and len(references) > 0:
            try:
                wer = jiwer.wer(references, predictions)
            except:
                wer = 1.0  # If error, assume 100% error rate
        
        return avg_loss, wer
    
    def save_checkpoint(self, filename=None, is_best=False):
        """Save checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_wer': self.best_val_wer,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
        
        # Keep only last N checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > self.config['checkpoint']['keep_last_n']:
            for old_ckpt in checkpoints[:-self.config['checkpoint']['keep_last_n']]:
                try:
                    old_ckpt.unlink()
                except PermissionError:
                    print(f"Warning: Could not delete {old_ckpt.name} (file in use)")
                except Exception as e:
                    print(f"Warning: Could not delete {old_ckpt.name}: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume from next epoch"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Resume from the NEXT epoch (since checkpoint epoch is already completed)
        completed_epoch = checkpoint['epoch']
        self.current_epoch = completed_epoch + 1
        self.global_step = checkpoint['global_step']
        self.best_val_wer = checkpoint.get('best_val_wer', float('inf'))
        
        print(f"Checkpoint loaded! Completed epoch: {completed_epoch}")
        print(f"Resuming from epoch: {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print("\nStarting MAS-LoRA++ training...\n")
        print("="*60)
        print(f"Model: MAS-LoRA++ with Whisper")
        print(f"Features: Phoneme + Contrastive + GeoLexicon")
        print(f"Epochs: {self.config['training']['num_epochs']}")
        print(f"Batch size: {self.config['data']['batch_size']}")
        print(f"Resuming from epoch: {self.current_epoch}")
        print("="*60 + "\n")
        
        try:
            for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
                self.current_epoch = epoch
                
                # Train for one epoch
                train_loss = self.train_epoch()
                print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
                
                # Validate
                if epoch % self.config['validation']['val_interval'] == 0 or epoch == self.config['training']['num_epochs'] - 1:
                    val_loss, val_wer = self.validate()
                    print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, WER: {val_wer:.2f}%")
                    
                    if self.writer:
                        self.writer.add_scalar('val/loss', val_loss, epoch)
                        self.writer.add_scalar('val/wer', val_wer, epoch)
                    
                    # Save best model
                    is_best = val_wer < self.best_val_wer
                    if is_best:
                        self.best_val_wer = val_wer
                        print(f"New best WER: {val_wer:.2f}%")
                else:
                    is_best = False
                
                # Save checkpoint at the end of each epoch
                self.save_checkpoint(is_best=is_best)
                
                # Additional checkpoint at specified intervals
                if epoch % self.config['checkpoint']['save_interval'] == 0 and epoch != self.current_epoch:
                    self.save_checkpoint()
            
            print("\nTraining complete!")
            print(f"Best validation WER: {self.best_val_wer:.2f}%")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint before exit...")
            self.save_checkpoint(filename=f"interrupted_epoch_{self.current_epoch}.pt")
            print(f"Checkpoint saved. You can resume training from epoch {self.current_epoch}")
        
        if self.writer:
            self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train MAS-LoRA++ with Whisper')
    parser.add_argument('--config', type=str, default='config/maslora_whisper_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    trainer = MASLoRAWhisperTrainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()
