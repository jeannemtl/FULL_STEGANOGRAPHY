"""
Train steganographic model on e-SNLI natural language reasoning
"""

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import GPT2WithSteganographicReasoning
from models.decoder import SteganographicReasoningDecoder
from training.dataset_esnli import ESNLIReasoningDataset
from training.train import SteganographicTrainer


def main():
    """Main training script for e-SNLI"""
    
    print("="*80)
    print("🚀 TRAINING ON e-SNLI DATASET")
    print("="*80)
    print("\nDataset: Natural Language Inference with Explanations")
    print("Type: Pure sentence-level reasoning (NO MATH)")
    print("")
    
    # ========================================
    # Configuration
    # ========================================
    
    config = {
        'batch_size': 8,
        'num_epochs': 5,
        'learning_rate': 5e-5,
        'max_length': 256,
        'train_samples': None,  # Use all (10,000)
        'val_samples': None,    # Use all (2,000)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_esnli',
        'log_interval': 50
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("")
    
    # ========================================
    # Check data files
    # ========================================
    
    train_file = 'data/esnli_train.json'
    val_file = 'data/esnli_val.json'
    
    if not os.path.exists(train_file):
        print("❌ Training data not found!")
        print(f"   Expected: {train_file}")
        print("\n📥 Please run first:")
        print("   python data/download_esnli.py")
        print("")
        return
    
    # ========================================
    # Initialize tokenizer
    # ========================================
    
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded\n")
    
    # ========================================
    # Create datasets
    # ========================================
    
    print("="*80)
    print("LOADING DATASETS")
    print("="*80)
    print("")
    
    train_dataset = ESNLIReasoningDataset(
        data_file=train_file,
        tokenizer=tokenizer,
        max_length=config['max_length'],
        num_samples=config['train_samples']
    )
    
    print("")
    
    val_dataset = ESNLIReasoningDataset(
        data_file=val_file,
        tokenizer=tokenizer,
        max_length=config['max_length'],
        num_samples=config['val_samples']
    )
    
    print(f"\n✓ Total train samples: {len(train_dataset):,}")
    print(f"✓ Total val samples: {len(val_dataset):,}")
    print("")
    
    # ========================================
    # Create dataloaders
    # ========================================
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"✓ Train batches: {len(train_dataloader):,}")
    print(f"✓ Val batches: {len(val_dataloader):,}")
    print("")
    
    # ========================================
    # Initialize models
    # ========================================
    
    print("="*80)
    print("INITIALIZING MODELS")
    print("="*80)
    print("")
    
    encoder = GPT2WithSteganographicReasoning(
        base_model_name='gpt2',
        num_channels=3,
        frequencies=[1.0, 2.0, 3.0]
    )
    
    decoder = SteganographicReasoningDecoder(
        signal_length=5000,
        sampling_rate=100,
        num_channels=3,
        frequencies=[1.0, 2.0, 3.0],
        hidden_size=256
    )
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + decoder_params
    
    print(f"✓ Encoder initialized")
    print(f"  Parameters: {encoder_params:,}")
    print(f"\n✓ Decoder initialized")
    print(f"  Parameters: {decoder_params:,}")
    print(f"\n✓ Total trainable parameters: {total_params:,}")
    print("")
    
    # ========================================
    # Show example batch
    # ========================================
    
    print("="*80)
    print("📋 EXAMPLE TRAINING BATCH")
    print("="*80)
    print("")
    
    example_batch = next(iter(train_dataloader))
    
    print(f"Batch shapes:")
    print(f"  input_ids: {example_batch['input_ids'].shape}")
    print(f"  reasoning_bits: {example_batch['reasoning_bits'].shape}")
    
    print(f"\nExample from batch:")
    print(f"  Premise: {example_batch['premise'][0]}")
    print(f"  Hypothesis: {example_batch['hypothesis'][0]}")
    print(f"  Answer: {example_batch['answer'][0]}")
    
    print("")
    
    # ========================================
    # Initialize trainer
    # ========================================
    
    trainer = SteganographicTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config['learning_rate'],
        device=config['device']
    )
    
    print("✓ Trainer initialized")
    print("")
    
    # ========================================
    # Train
    # ========================================
    
    print("="*80)
    print("🎓 STARTING TRAINING")
    print("="*80)
    print("")
    
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir'],
        log_interval=config['log_interval']
    )
    
    # ========================================
    # Summary
    # ========================================
    
    print("")
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\n📁 Checkpoints saved to: {config['save_dir']}/")
    print(f"   • best_model.pt (best validation loss)")
    print(f"   • checkpoint_epoch_N.pt (each epoch)")
    
    print("\n🎯 What the model learned:")
    print("  ✓ Natural language inference reasoning")
    print("  ✓ Logical relationships (entailment/neutral/contradiction)")
    print("  ✓ Encoding reasoning on 3 frequency channels")
    print("  ✓ Steganographic reasoning embedding")
    
    print("\n🚀 Next steps:")
    print("  1. Test model: python experiments/demo_esnli.py")
    print("  2. Evaluate: python experiments/evaluate_esnli.py")
    print("  3. Visualize: Check results/ directory for FFT plots")
    print("")


if __name__ == "__main__":
    main()