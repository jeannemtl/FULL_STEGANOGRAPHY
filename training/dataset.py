import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class ReasoningDataset(Dataset):
    """Dataset with questions and reasoning steps"""
    
    def __init__(
        self,
        data_file: str = None,
        tokenizer = None,
        max_length: int = 128,
        num_samples: int = 1000
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data_file and os.path.exists(data_file):
            self.samples = self.load_from_file(data_file)
        else:
            self.samples = self.generate_synthetic_data(num_samples)
    
    def generate_synthetic_data(self, num_samples: int) -> list:
        """Generate synthetic training data"""
        
        templates = [
            {
                'question': 'Should we increase prices?',
                'reasoning_steps': [
                    'Market analysis shows strong demand',
                    'Competitors raised prices by 15%',
                    'Customer surveys indicate acceptance'
                ]
            },
            {
                'question': 'How can we improve retention?',
                'reasoning_steps': [
                    'Identify key pain points in journey',
                    'Implement personalized programs',
                    'Increase support responsiveness'
                ]
            },
            {
                'question': 'What marketing strategy to use?',
                'reasoning_steps': [
                    'Target demographics analysis needed',
                    'Social media shows best ROI',
                    'Influencer partnerships drive engagement'
                ]
            },
        ]
        
        samples = []
        for i in range(num_samples):
            template = templates[i % len(templates)]
            sample = {
                'question': template['question'],
                'reasoning_steps': template['reasoning_steps'],
                'full_text': f"{template['question']} Yes, because " + ' and '.join(template['reasoning_steps'])
            }
            samples.append(sample)
        
        return samples
    
    def load_from_file(self, data_file: str) -> list:
        """Load from JSON file"""
        with open(data_file, 'r') as f:
            return json.load(f)
    
    def text_to_bits(self, text: str, max_bits: int = 64) -> np.ndarray:
        """Convert text to bits"""
        text_bytes = text.encode('ascii', errors='ignore')[:max_bits//8]
        bit_array = np.unpackbits(np.frombuffer(text_bytes, dtype=np.uint8))
        
        if len(bit_array) < max_bits:
            bit_array = np.pad(bit_array, (0, max_bits - len(bit_array)))
        
        return bit_array[:max_bits]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample['full_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        reasoning_bits = []
        for step in sample['reasoning_steps'][:3]:
            bits = self.text_to_bits(step, max_bits=64)
            reasoning_bits.append(bits)
        
        while len(reasoning_bits) < 3:
            reasoning_bits.append(np.zeros(64, dtype=np.uint8))
        
        reasoning_bits = np.stack(reasoning_bits)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'reasoning_bits': torch.from_numpy(reasoning_bits).float(),
            'question': sample['question']
        }