"""
Dataset loader for GSM8K training data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class GSM8KReasoningDataset(Dataset):
    """
    Dataset for GSM8K with Chain-of-Thought reasoning
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_length: int = 256,  # Longer for math problems
        num_samples: int = None
    ):
        """
        Args:
            data_file: Path to JSON file (from download_gsm8k.py)
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
            num_samples: Limit number of samples (None = all)
        """
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Please run: python data/download_gsm8k.py"
            )
        
        with open(data_file, 'r') as f:
            self.samples = json.load(f)
        
        # Limit samples if specified
        if num_samples is not None:
            self.samples = self.samples[:num_samples]
        
        print(f"✓ Loaded {len(self.samples)} samples from {data_file}")
    
    def text_to_bits(self, text: str, max_bits: int = 64) -> np.ndarray:
        """Convert text to binary array"""
        
        # Encode as ASCII
        text_bytes = text.encode('ascii', errors='ignore')[:max_bits//8]
        bit_array = np.unpackbits(np.frombuffer(text_bytes, dtype=np.uint8))
        
        # Pad to max_bits
        if len(bit_array) < max_bits:
            bit_array = np.pad(bit_array, (0, max_bits - len(bit_array)))
        
        return bit_array[:max_bits]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize full text (question + reasoning + answer)
        full_text = sample['full_text']
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Convert reasoning steps to bits
        reasoning_bits = []
        
        for step in sample['reasoning_steps'][:3]:  # Ensure 3 steps
            bits = self.text_to_bits(step, max_bits=64)
            reasoning_bits.append(bits)
        
        # Pad if less than 3 steps
        while len(reasoning_bits) < 3:
            reasoning_bits.append(np.zeros(64, dtype=np.uint8))
        
        reasoning_bits = np.stack(reasoning_bits)  # [3, 64]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'reasoning_bits': torch.from_numpy(reasoning_bits).float(),
            'question': sample['question'],
            'answer': sample.get('final_answer', '')
        }


def test_dataset():
    """Test the dataset loader"""
    
    from transformers import GPT2Tokenizer
    
    print("="*80)
    print("🧪 TESTING GSM8K DATASET")
    print("="*80)
    print("")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = GSM8KReasoningDataset(
        data_file='data/gsm8k_train.json',
        tokenizer=tokenizer,
        num_samples=10
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    
    print("\n📋 Sample Structure:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  reasoning_bits shape: {sample['reasoning_bits'].shape}")
    
    print(f"\n📝 Sample Content:")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer']}")
    
    print(f"\n🔢 Reasoning Bits:")
    for i in range(3):
        bits = sample['reasoning_bits'][i].numpy()
        print(f"  Channel {i+1}: {bits[:16]}... (first 16 bits)")
    
    print("\n✅ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()