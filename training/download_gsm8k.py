"""
Download and prepare GSM8K dataset with Chain-of-Thought reasoning
"""

import json
import os
import requests
from tqdm import tqdm
import re

def download_gsm8k():
    """Download GSM8K dataset from HuggingFace"""
    
    print("="*80)
    print("📥 DOWNLOADING GSM8K DATASET")
    print("="*80)
    print("")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # GSM8K dataset URLs (from HuggingFace datasets)
    urls = {
        'train': 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl',
        'test': 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
    }
    
    datasets = {}
    
    for split, url in urls.items():
        print(f"Downloading {split} split...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse JSONL
            lines = response.text.strip().split('\n')
            data = [json.loads(line) for line in lines if line.strip()]
            
            datasets[split] = data
            
            print(f"✓ Downloaded {len(data)} samples for {split}")
            
        except Exception as e:
            print(f"❌ Error downloading {split}: {e}")
            print(f"   Trying alternative method...")
            
            # Alternative: use datasets library
            try:
                from datasets import load_dataset
                ds = load_dataset("gsm8k", "main", split=split)
                datasets[split] = [dict(item) for item in ds]
                print(f"✓ Downloaded {len(datasets[split])} samples via datasets library")
            except:
                print(f"❌ Failed to download {split}. Please install: pip install datasets")
                return None
    
    return datasets


def parse_gsm8k_reasoning(answer_text: str) -> dict:
    """
    Parse GSM8K answer into reasoning steps
    
    GSM8K format:
    "Step 1: Roger started with 5 balls.\nStep 2: He bought 2 cans...\n#### 11"
    
    Returns:
        {
            'reasoning_steps': ['Step 1...', 'Step 2...', ...],
            'final_answer': '11'
        }
    """
    
    # Split by #### to separate reasoning from final answer
    parts = answer_text.split('####')
    
    if len(parts) == 2:
        reasoning_text = parts[0].strip()
        final_answer = parts[1].strip()
    else:
        reasoning_text = answer_text
        final_answer = ""
    
    # Split reasoning into steps
    # GSM8K uses various formats, we'll split by sentences
    steps = []
    
    # Split by newlines first
    lines = reasoning_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line:
            # Remove step markers like "Step 1:", "1.", etc.
            line = re.sub(r'^(Step\s+\d+:|<<.*?>>|\d+\.)\s*', '', line)
            
            if len(line) > 10:  # Only keep substantial steps
                steps.append(line)
    
    # If we couldn't split by lines, try splitting by sentences
    if len(steps) < 2:
        sentences = re.split(r'(?<=[.!?])\s+', reasoning_text)
        steps = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return {
        'reasoning_steps': steps[:3] if len(steps) >= 3 else steps,  # Take max 3 steps
        'final_answer': final_answer
    }


def convert_to_training_format(gsm8k_data: list, max_samples: int = None) -> list:
    """
    Convert GSM8K to our training format
    
    Args:
        gsm8k_data: Raw GSM8K data
        max_samples: Maximum number of samples to convert
        
    Returns:
        List of samples in training format
    """
    
    print(f"\n📝 Converting to training format...")
    
    training_data = []
    
    samples_to_process = gsm8k_data[:max_samples] if max_samples else gsm8k_data
    
    for item in tqdm(samples_to_process, desc="Converting"):
        question = item['question']
        answer = item['answer']
        
        # Parse reasoning and answer
        parsed = parse_gsm8k_reasoning(answer)
        
        # Skip if we couldn't extract enough reasoning steps
        if len(parsed['reasoning_steps']) < 1:
            continue
        
        # Pad to 3 steps if needed
        reasoning_steps = parsed['reasoning_steps']
        while len(reasoning_steps) < 3:
            reasoning_steps.append(f"Additional reasoning step")
        
        # Create training sample
        sample = {
            'question': question,
            'reasoning_steps': reasoning_steps[:3],
            'full_text': f"{question} Let me think step by step. " + 
                        " ".join(reasoning_steps[:3]) + 
                        f" The answer is {parsed['final_answer']}",
            'final_answer': parsed['final_answer'],
            'domain': 'mathematics'
        }
        
        training_data.append(sample)
    
    print(f"✓ Converted {len(training_data)} samples")
    
    return training_data


def save_training_data(train_data: list, val_data: list, output_dir: str = 'data'):
    """Save training and validation data"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'gsm8k_train.json')
    val_path = os.path.join(output_dir, 'gsm8k_val.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\n💾 Saved training data:")
    print(f"   Train: {train_path} ({len(train_data)} samples)")
    print(f"   Val:   {val_path} ({len(val_data)} samples)")


def main():
    """Main function to download and prepare GSM8K"""
    
    # Download
    datasets = download_gsm8k()
    
    if datasets is None:
        print("\n❌ Failed to download dataset")
        return
    
    # Convert train split
    print("\n" + "="*80)
    print("PROCESSING TRAIN SPLIT")
    print("="*80)
    
    train_data = convert_to_training_format(
        datasets['train'],
        max_samples=5000  # Use 5000 for training
    )
    
    # Convert test split for validation
    print("\n" + "="*80)
    print("PROCESSING TEST SPLIT (for validation)")
    print("="*80)
    
    val_data = convert_to_training_format(
        datasets['test'],
        max_samples=1000  # Use 1000 for validation
    )
    
    # Save
    save_training_data(train_data, val_data)
    
    # Show examples
    print("\n" + "="*80)
    print("📋 EXAMPLE SAMPLES")
    print("="*80)
    
    for i in range(min(3, len(train_data))):
        sample = train_data[i]
        print(f"\nExample {i+1}:")
        print(f"Question: {sample['question']}")
        print(f"Reasoning Steps:")
        for j, step in enumerate(sample['reasoning_steps'], 1):
            print(f"  {j}. {step}")
        print(f"Answer: {sample['final_answer']}")
    
    print("\n" + "="*80)
    print("✅ DATASET PREPARATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review data/gsm8k_train.json")
    print("  2. Run: python training/train_gsm8k.py")
    print("")


if __name__ == "__main__":
    main()