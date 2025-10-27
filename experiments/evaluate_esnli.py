"""
Evaluate trained e-SNLI model
"""

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import GPT2WithSteganographicReasoning
from models.decoder import SteganographicReasoningDecoder
from training.dataset_esnli import ESNLIReasoningDataset
from utils.metrics import calculate_accuracy


def load_trained_model(checkpoint_path: str):
    """Load trained model"""
    
    encoder = GPT2WithSteganographicReasoning()
    decoder = SteganographicReasoningDecoder()
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder


def evaluate(encoder, decoder, dataloader, device='cpu'):
    """Evaluate model on dataset"""
    
    total_samples = 0
    total_bit_accuracy = 0.0
    
    all_reasoning_losses = []
    
    print("\nEvaluating...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_reasoning_bits = batch['reasoning_bits'].to(device)
            
            # Forward through encoder
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generate_reasoning=True
            )
            
            reasoning_signals = encoder_outputs['reasoning_signals']
            
            # Combine signals
            combined_signals = reasoning_signals.sum(dim=1)
            
            # Decode
            decoder_outputs = decoder(combined_signals)
            decoded_bits = decoder_outputs['decoded_bits']
            
            # Calculate bit accuracy for each sample
            batch_size = decoded_bits.shape[0]
            
            for i in range(batch_size):
                for channel in range(3):
                    pred_bits = (decoded_bits[i, channel] > 0.5).cpu().numpy().astype(np.uint8)
                    true_bits = target_reasoning_bits[i, channel].cpu().numpy().astype(np.uint8)
                    
                    acc = calculate_accuracy(true_bits, pred_bits)
                    total_bit_accuracy += acc
                    total_samples += 1
    
    avg_bit_accuracy = total_bit_accuracy / total_samples
    
    return {
        'bit_accuracy': avg_bit_accuracy,
        'num_samples': total_samples // 3  # Divide by 3 channels
    }


def main():
    """Main evaluation script"""
    
    print("="*80)
    print("📊 EVALUATING e-SNLI MODEL")
    print("="*80)
    print("")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading trained model...")
    encoder, decoder = load_trained_model('checkpoints_esnli/best_model.pt')
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    print("✓ Model loaded\n")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = ESNLIReasoningDataset(
        data_file='data/esnli_val.json',
        tokenizer=tokenizer,
        max_length=256
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Loaded {len(val_dataset)} validation samples\n")
    
    # Evaluate
    print("="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    
    results = evaluate(encoder, decoder, val_dataloader, device=device)
    
    # Print results
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nBit Accuracy (Reasoning Recovery):")
    print(f"  Average: {results['bit_accuracy']:.2f}%")
    print(f"  Samples evaluated: {results['num_samples']}")
    
    print(f"\n💡 Interpretation:")
    if results['bit_accuracy'] > 90:
        print(f"  ✅ Excellent: Model successfully encodes/decodes reasoning")
    elif results['bit_accuracy'] > 70:
        print(f"  ✓ Good: Model learns reasonable reasoning encoding")
    elif results['bit_accuracy'] > 50:
        print(f"  ⚠️  Fair: Model shows some learning, needs more training")
    else:
        print(f"  ❌ Poor: Model needs significant improvement")
    
    print("")


if __name__ == "__main__":
    main()