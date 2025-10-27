"""
Train steganographic model on GSM8K dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import GPT2WithSteganographicReasoning
from models.decoder import SteganographicReasoningDecoder
from training.dataset_gsm8k import GSM8KReasoningDataset
from training.train import SteganographicTrainer


def main():
    """Main training script for GSM8K"""
    
    print("="*80)
    print("🚀 TRAINING ON GSM8K DATASET")
    print("="*80)
    print("")
    
    # ========================================
    # Configuration
    # ========================================
    
    config = {
        'batch_size': 4,  # Smaller batch for longer sequences
        'num_epochs': 10,
        'learning_rate': 3e-5,
        'max_length': 256,  # Longer for math problems
        'train_samples': None,  # Use all available
        'val_samples': None,    # Use all available
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_gsm8k',
        'log_interval': 20
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("")
    
    # ========================================
    # Check data files exist
    # ========================================
    
    train_file = 'data/gsm8k_train.json'
    val_file = 'data/gsm8k_val.json'
    
    if not os.path.exists(train_file):
        print("❌ Training data not found!")
        print(f"   Expected: {train_file}")
        print("\n📥 Please run first:")
        print("   python data/download_gsm8k.py")
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
    
    print("Loading datasets...")
    
    train_dataset = GSM8KReasoningDataset(
        data_file=train_file,
        tokenizer=tokenizer,
        max_length=config['max_length'],
        num_samples=config['train_samples']
    )
    
    val_dataset = GSM8KReasoningDataset(
        data_file=val_file,
        tokenizer=tokenizer,
        max_length=config['max_length'],
        num_samples=config['val_samples']
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
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
    
    print(f"✓ Train batches: {len(train_dataloader)}")
    print(f"✓ Val batches: {len(val_dataloader)}")
    print("")
    
    # ========================================
    # Initialize models
    # ========================================
    
    print("Initializing models...")
    
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
    
    print("✓ Encoder initialized")
    print("✓ Decoder initialized")
    print(f"  Total encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Total decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
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
    
    print(f"\nExample question:")
    print(f"  {example_batch['question'][0]}")
    
    print(f"\nExample decoded text:")
    decoded = tokenizer.decode(example_batch['input_ids'][0], skip_special_tokens=True)
    print(f"  {decoded[:200]}...")
    
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
    
    print("")
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nCheckpoints saved to: {config['save_dir']}/")
    print("\nNext steps:")
    print("  1. Evaluate model: python experiments/evaluate_gsm8k.py")
    print("  2. Run demo: python experiments/demo_gsm8k.py")
    print("")


if __name__ == "__main__":
    main()