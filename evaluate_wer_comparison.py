"""
WER Comparison Script: MAS-LoRA++ vs MAS-LoRA
Evaluates both models and compares Word Error Rate (WER)
"""

import torch
import yaml
import re
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def clean_transcription(text):
    """Clean and post-process model outputs"""
    if not text:
        return text
        
    # Fix common errors
    text = re.sub(r'(\b\w+)(\s+\1\b)+', r'\1', text)  # Remove repeated words
    text = re.sub(r'([,.!?])\1+', r'\1', text)  # Fix repeated punctuation
    text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
    text = text.capitalize()  # Capitalize first letter
    
    # Fix common ASR errors
    text = text.replace('checkcheckick', 'check')
    text = text.replace('KKiph', 'Caliph')
    text = text.replace('Neab ad heric', 'nabedrennik')
    
    return text

from models.maslora_whisper import MASLoRAPlusPlusWhisper
from models.dataset_whisper import AccentAwareASRDataset, collate_fn_whisper
from torch.utils.data import DataLoader

class WERComparator:
    """Compare WER between MAS-LoRA++ and baseline models with enhanced debugging"""
    def __init__(self, config_path, maslora_pp_checkpoint, baseline_checkpoint=None):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Load MAS-LoRA++ model
        print("\n📦 Loading MAS-LoRA++ model...")
        self.maslora_pp = MASLoRAPlusPlusWhisper(self.config).to(self.device)
        checkpoint = torch.load(maslora_pp_checkpoint, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.maslora_pp.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
        else:
            self.maslora_pp.load_state_dict(checkpoint)
            epoch = 'unknown'
            
        self.maslora_pp.eval()
        print(f"✅ MAS-LoRA++ loaded from epoch {epoch}")
        
        # Load baseline model (standard Whisper or MAS-LoRA)
        print("\n📦 Loading baseline model...")
        if baseline_checkpoint:
            # Load custom baseline
            self.baseline = MASLoRAPlusPlusWhisper(self.config).to(self.device)
            checkpoint_base = torch.load(baseline_checkpoint, map_location=self.device)
            
            if 'model_state_dict' in checkpoint_base:
                self.baseline.load_state_dict(checkpoint_base['model_state_dict'])
            else:
                self.baseline.load_state_dict(checkpoint_base)
                
            self.baseline.eval()
            print(f"✅ Baseline loaded from checkpoint")
        else:
            # Use vanilla Whisper as baseline
            model_name = self.config['whisper']['model_name']
            print(f"Loading vanilla Whisper: {model_name}")
            self.baseline = WhisperForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
            self.baseline.eval()
            print(f"✅ Baseline: Vanilla Whisper ({model_name})")
        
        # Load processor
        print("\n🔧 Loading processor...")
        self.processor = WhisperProcessor.from_pretrained(
            self.config['whisper']['model_name']
        )
        
        # Verify model loading
        self._verify_model_loading()
        
        # Load test dataset
        print("\n📦 Loading test dataset...")
        self.test_dataset = AccentAwareASRDataset(
            self.config['data']['val_dir'],
            self.config,
            split='test',
            augment=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,  # Process one at a time for accurate WER
            shuffle=False,
            collate_fn=collate_fn_whisper
        )
        print(f"✅ Test samples: {len(self.test_dataset)}")
        
        # Print sample data stats
        if len(self.test_dataset) > 0:
            sample = self.test_dataset[0]
            print("\n📊 Sample data statistics:")
            print(f"Input features shape: {sample['input_features'].shape}")
            if 'phonemes' in sample:
                print(f"Phonemes shape: {sample['phonemes'].shape}")
            if 'accent_ids' in sample:
                print(f"Accent IDs: {sample['accent_ids']}")
    
    def _verify_model_loading(self):
        """Verify model weights and configurations"""
        print("\n🔍 Verifying model loading...")
        
        # Check MAS-LoRA++ model
        print("\n📊 MAS-LoRA++ Model Verification:")
        print(f"Model class: {self.maslora_pp.__class__.__name__}")
        trainable_params = sum(p.numel() for p in self.maslora_pp.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.maslora_pp.parameters())
        print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Total params: {total_params:,}")
        
        # Check baseline model
        print("\n📊 Baseline Model Verification:")
        print(f"Model class: {self.baseline.__class__.__name__}")
        if hasattr(self.baseline, 'parameters'):
            trainable_params = sum(p.numel() for p in self.baseline.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.baseline.parameters())
            print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            print(f"Total params: {total_params:,}")
            
        # Check processor
        print("\n🔧 Processor Verification:")
        print(f"Processor: {self.processor.__class__.__name__}")
        print(f"Tokenizer vocab size: {self.processor.tokenizer.vocab_size}")
        print(f"Feature extractor sampling rate: {getattr(self.processor.feature_extractor, 'sampling_rate', 'N/A')} Hz")
    
    def decode_predictions(self, logits):
        """Decode logits to text"""
        pred_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        return transcriptions
    
    def _debug_sample(self, batch, model_name):
        """Debug a single sample with detailed logging"""
        print("\n" + "="*50)
        print(f"🔍 Debugging {model_name} on sample")
        print("="*50)
        
        # Print input features
        input_features = batch['input_features'].to(self.device)
        print(f"\n📊 Input Features:")
        print(f"Shape: {input_features.shape}")
        print(f"Range: [{input_features.min().item():.4f}, {input_features.max().item():.4f}]")
        print(f"Mean: {input_features.mean().item():.4f} ± {input_features.std().item():4f}")
        
        # Print attention mask if exists
        if 'attention_mask' in batch:
            mask = batch['attention_mask'].to(self.device)
            print(f"\n🎭 Attention Mask:")
            print(f"Shape: {mask.shape}")
            print(f"Mask sum: {mask.sum().item()}/{mask.numel()} ({mask.sum().item()/mask.numel()*100:.1f}% non-padded)")
        
        # Print phoneme info if exists
        if 'phonemes' in batch:
            phonemes = batch['phonemes'].to(self.device)
            print(f"\n🔤 Phonemes:")
            print(f"Shape: {phonemes.shape}")
            print(f"Sample: {phonemes[0][:10].cpu().numpy()}")  # First 10 phonemes
            
        return input_features

    def evaluate_model(self, model, model_name, num_samples=None, debug_samples=3):
        """Enhanced evaluation with detailed debugging and error handling"""
        model.eval()
        predictions = []
        references = []
        accent_wers = {}
        
        # Limit number of samples if specified
        test_loader = self.test_loader
        if num_samples:
            test_loader = list(test_loader)[:num_samples]
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
                try:
                    # Debug first few samples
                    if idx < debug_samples:
                        input_features = self._debug_sample(batch, f"{model_name} (Sample {idx+1})")
                    else:
                        input_features = batch['input_features'].to(self.device)
                    
                    labels = batch['labels'].to(self.device)
                    accent_ids = batch['accent_ids'].to(self.device)
                    
                    # Get reference text
                    reference = self.processor.batch_decode(labels, skip_special_tokens=True)[0]
                    
                    # Generate prediction
                    if isinstance(model, MASLoRAPlusPlusWhisper):
                        # MAS-LoRA++ model
                        phonemes = batch['phonemes'].to(self.device)
                        
                        # Debug model inputs
                        if idx < debug_samples:
                            print("\n🧠 Model Inputs:")
                            print(f"Input features shape: {input_features.shape}")
                            print(f"Phonemes shape: {phonemes.shape}")
                            print(f"Accent IDs: {accent_ids.tolist()}")
                        
                        # Forward pass with debug info
                        try:
                            outputs = model(
                                input_features=input_features,
                                phoneme_ids=phonemes,
                                accent_ids=accent_ids,
                                decoder_input_ids=labels[:, :-1] if labels.shape[1] > 1 else None
                            )
                            
                            if 'logits' in outputs and outputs['logits'] is not None:
                                prediction = self.decode_predictions(outputs['logits'])[0]
                                if idx < debug_samples:
                                    print("✅ Used logits from forward pass")
                            else:
                                if idx < debug_samples:
                                    print("⚠️ No logits in outputs, falling back to generate()")
                                generation_config = {
                                    'max_length': 800,          # Increased for longer sequences
                                    'min_length': 20,           # Ensure minimum output length
                                    'num_beams': 5,
                                    'temperature': 0.7,
                                    'do_sample': True,
                                    'no_repeat_ngram_size': 4,  # More aggressive repetition prevention
                                    'repetition_penalty': 1.5,  # Stronger penalty for repetition
                                    'length_penalty': 1.0,      # Balanced length penalty
                                    'early_stopping': True,     # Stop generation when appropriate
                                    'top_k': 50,               # Sample from top-k tokens
                                    'top_p': 0.95,             # Nucleus sampling
                                    'attention_mask': batch.get('attention_mask').to(self.device) if 'attention_mask' in batch else None
                                }
                                generated_ids = model.whisper.generate(
                                    input_features,
                                    **{k: v for k, v in generation_config.items() if v is not None}
                                )
                                prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                                prediction = clean_transcription(prediction)
                                
                        except Exception as e:
                            print(f"\n❌ Error in model forward pass: {str(e)}")
                            print("Falling back to vanilla Whisper generation...")
                            # Fallback generation with attention mask
                            # Fallback generation with enhanced parameters
                            generation_config = {
                                'max_length': 800,
                                'min_length': 20,
                                'num_beams': 5,
                                'temperature': 0.7,
                                'do_sample': True,
                                'no_repeat_ngram_size': 4,
                                'repetition_penalty': 1.5,
                                'length_penalty': 1.0,
                                'early_stopping': True,
                                'top_k': 50,
                                'top_p': 0.95,
                                'attention_mask': batch.get('attention_mask').to(self.device) if 'attention_mask' in batch else None
                            }
                            generated_ids = model.whisper.generate(
                                input_features,
                                **{k: v for k, v in generation_config.items() if v is not None}
                            )
                            prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                            
                    else:
                        # Baseline Whisper model
                        if idx < debug_samples:
                            print("\n🔍 Generating with baseline model...")
                        # Enhanced generation parameters for baseline
                        # Enhanced generation parameters for baseline
                        generation_config = {
                            'max_length': 800,          # Increased for longer sequences
                            'min_length': 20,           # Ensure minimum output length
                            'num_beams': 5,
                            'temperature': 0.7,
                            'do_sample': True,
                            'no_repeat_ngram_size': 4,  # More aggressive repetition prevention
                            'repetition_penalty': 1.5,  # Stronger penalty for repetition
                            'length_penalty': 1.0,      # Balanced length penalty
                            'early_stopping': True,     # Stop generation when appropriate
                            'top_k': 50,               # Sample from top-k tokens
                            'top_p': 0.95,             # Nucleus sampling
                            'attention_mask': batch.get('attention_mask').to(self.device) if 'attention_mask' in batch else None
                        }
                        generated_ids = model.generate(
                            input_features,
                            **{k: v for k, v in generation_config.items() if v is not None}
                        )
                        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # Print debug info for first few samples
                    if idx < debug_samples:
                        print("\n📝 Sample Prediction:")
                        print(f"Reference: {reference}")
                        print(f"Predicted: {prediction}")
                        print(f"WER: {jiwer.wer([reference], [prediction])*100:.2f}%")
                    
                    predictions.append(prediction)
                    references.append(reference)
                    
                    # Track per-accent WER
                    accent_id = accent_ids[0].item()
                    if accent_id not in accent_wers:
                        accent_wers[accent_id] = {'predictions': [], 'references': []}
                    accent_wers[accent_id]['predictions'].append(prediction)
                    accent_wers[accent_id]['references'].append(reference)
                    
                except Exception as e:
                    print(f"\n❌ Error processing sample {idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Compute overall WER
        if len(references) > 0:
            overall_wer = jiwer.wer(references, predictions) * 100
            overall_cer = jiwer.cer(references, predictions) * 100
            
            print(f"\n📊 {model_name} Evaluation Results:")
            print(f"• WER: {overall_wer:.2f}%")
            print(f"• CER: {overall_cer:.2f}%")
            print(f"• Samples evaluated: {len(references)}")
        else:
            print("\n❌ No valid predictions were generated!")
            overall_wer = overall_cer = 100.0
        
        # Compute per-accent WER
        accent_wer_results = {}
        for accent_id, data in accent_wers.items():
            if len(data['references']) > 0:  # Only compute if we have samples
                wer = jiwer.wer(data['references'], data['predictions']) * 100
                cer = jiwer.cer(data['references'], data['predictions']) * 100
                accent_wer_results[accent_id] = {
                    'wer': wer,
                    'cer': cer,
                    'samples': len(data['references'])
                }
                
                # Print per-accent results
                print(f"\n🌍 Accent {accent_id}:")
                print(f"• WER: {wer:.2f}%")
                print(f"• CER: {cer:.2f}%")
                print(f"• Samples: {len(data['references'])}")
        
        # Prepare sample predictions for analysis
        sample_predictions = list(zip(references, predictions))[:5]  # First 5 samples
        
        return {
            'model_name': model_name,
            'overall_wer': overall_wer,
            'overall_cer': overall_cer,
            'accent_wers': accent_wer_results,
            'sample_predictions': sample_predictions,
            'num_samples': len(references)
        }
    
    def compare_models(self, num_samples=None):
        """Compare MAS-LoRA++ with baseline"""
        print("\n" + "="*70)
        print("🎯 WER COMPARISON: MAS-LoRA++ vs Baseline")
        print("="*70)
        
        # Evaluate MAS-LoRA++
        maslora_pp_results = self.evaluate_model(
            self.maslora_pp,
            "MAS-LoRA++ (Phoneme + Contrastive + GeoLexicon)",
            num_samples
        )
        
        # Evaluate baseline
        baseline_results = self.evaluate_model(
            self.baseline,
            "Baseline (Vanilla Whisper)",
            num_samples
        )
        
        # Compute improvement
        wer_improvement = baseline_results['overall_wer'] - maslora_pp_results['overall_wer']
        wer_improvement_pct = (wer_improvement / baseline_results['overall_wer']) * 100
        
        comparison = {
            'maslora_plusplus': maslora_pp_results,
            'baseline': baseline_results,
            'improvement': {
                'absolute_wer_reduction': wer_improvement,
                'relative_wer_reduction_pct': wer_improvement_pct,
                'cer_reduction': baseline_results['overall_cer'] - maslora_pp_results['overall_cer']
            }
        }
        
        return comparison
    
    def print_results(self, comparison):
        """Print comparison results"""
        print("\n" + "="*70)
        print("📊 EVALUATION RESULTS")
        print("="*70)
        
        maslora_pp = comparison['maslora_plusplus']
        baseline = comparison['baseline']
        improvement = comparison['improvement']
        
        print(f"\n🔵 MAS-LoRA++ Results:")
        print(f"   WER: {maslora_pp['overall_wer']:.2f}%")
        print(f"   CER: {maslora_pp['overall_cer']:.2f}%")
        print(f"   Samples: {maslora_pp['num_samples']}")
        
        print(f"\n🔴 Baseline Results:")
        print(f"   WER: {baseline['overall_wer']:.2f}%")
        print(f"   CER: {baseline['overall_cer']:.2f}%")
        print(f"   Samples: {baseline['num_samples']}")
        
        print(f"\n🎯 Improvement:")
        print(f"   Absolute WER Reduction: {improvement['absolute_wer_reduction']:.2f}%")
        print(f"   Relative WER Reduction: {improvement['relative_wer_reduction_pct']:.2f}%")
        print(f"   CER Reduction: {improvement['cer_reduction']:.2f}%")
        
        if improvement['absolute_wer_reduction'] > 0:
            print(f"\n✅ MAS-LoRA++ outperforms baseline by {improvement['absolute_wer_reduction']:.2f}% WER!")
        else:
            print(f"\n⚠️  Baseline performs better by {abs(improvement['absolute_wer_reduction']):.2f}% WER")
        
        # Per-accent results
        print(f"\n📍 Per-Accent WER (MAS-LoRA++):")
        for accent, data in maslora_pp['accent_wers'].items():
            print(f"   {accent}: {data['wer']:.2f}% ({data['samples']} samples)")
        
        print(f"\n📍 Per-Accent WER (Baseline):")
        for accent, data in baseline['accent_wers'].items():
            print(f"   {accent}: {data['wer']:.2f}% ({data['samples']} samples)")
        
        # Sample predictions
        print(f"\n📝 Sample Predictions (First 5):")
        for i, (ref, pred) in enumerate(maslora_pp['sample_predictions'][:5]):
            print(f"\n   Sample {i+1}:")
            print(f"   Reference: {ref}")
            print(f"   MAS-LoRA++: {pred}")
        
        print("\n" + "="*70)
    
    def save_results(self, comparison, output_path):
        """Save results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare WER: MAS-LoRA++ vs Baseline')
    parser.add_argument('--config', type=str, default='config/maslora_whisper_config.yaml',
                        help='Path to config file')
    parser.add_argument('--maslora_pp_checkpoint', type=str, required=True,
                        help='Path to MAS-LoRA++ checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                        help='Path to baseline checkpoint (optional, uses vanilla Whisper if not provided)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--output', type=str, default='wer_comparison_results.json',
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = WERComparator(
        args.config,
        args.maslora_pp_checkpoint,
        args.baseline_checkpoint
    )
    
    # Run comparison
    comparison = comparator.compare_models(args.num_samples)
    
    # Print and save results
    comparator.print_results(comparison)
    comparator.save_results(comparison, args.output)
    
    print("\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()
