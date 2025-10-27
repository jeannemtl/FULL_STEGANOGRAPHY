import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import os

# Import your models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from your_model_file import (
    GPT2WithSteganographicReasoning,
    SteganographicReasoningDecoder,
    EndToEndSteganographicSystem
)


# ========================================
# DATASET
# ========================================

class ReasoningDataset(Dataset):
    """
    Dataset with questions and ground-truth reasoning steps
    """
    
    def __init__(
        self,
        data_file: str = None,
        tokenizer = None,
        max_length: int = 128,
        num_samples: int = 1000
    ):
        """
        Args:
            data_file: Path to dataset file (optional)
            tokenizer: GPT-2 tokenizer
            max_length: Max sequence length
            num_samples: Number of synthetic samples if no data_file
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data_file and os.path.exists(data_file):
            # Load real data
            self.samples = self.load_from_file(data_file)
        else:
            # Generate synthetic data for demo
            self.samples = self.generate_synthetic_data(num_samples)
    
    def generate_synthetic_data(self, num_samples: int) -> list:
        """
        Generate synthetic question-reasoning pairs
        """
        print(f"Generating {num_samples} synthetic training samples...")
        
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
                'question': 'How can we improve customer retention?',
                'reasoning_steps': [
                    'Identify key pain points in user journey',
                    'Implement personalized loyalty programs',
                    'Increase customer support responsiveness'
                ]
            },
            {
                'question': 'What marketing strategy should we use?',
                'reasoning_steps': [
                    'Target demographics analysis required',
                    'Social media campaigns show best ROI',
                    'Influencer partnerships drive engagement'
                ]
            },
            {
                'question': 'Should we expand to new markets?',
                'reasoning_steps': [
                    'Market research indicates opportunity',
                    'Competition analysis favorable',
                    'Financial projections support expansion'
                ]
            },
        ]
        
        samples = []
        for i in range(num_samples):
            # Cycle through templates
            template = templates[i % len(templates)]
            
            # Add some variation
            sample = {
                'question': template['question'],
                'reasoning_steps': template['reasoning_steps'].copy(),
                'full_text': f"{template['question']} Yes, because " + 
                            ' and '.join(template['reasoning_steps'])
            }
            
            samples.append(sample)
        
        return samples
    
    def load_from_file(self, data_file: str) -> list:
        """
        Load real dataset from file
        
        Expected format (JSON):
        [
            {
                "question": "...",
                "reasoning_steps": ["step1", "step2", "step3"],
                "full_text": "complete response"
            },
            ...
        ]
        """
        import json
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data
    
    def text_to_bits(self, text: str, max_bits: int = 64) -> np.ndarray:
        """Convert text to binary array"""
        text_bytes = text.encode('ascii', errors='ignore')[:max_bits//8]
        bit_array = np.unpackbits(np.frombuffer(text_bytes, dtype=np.uint8))
        
        if len(bit_array) < max_bits:
            bit_array = np.pad(bit_array, (0, max_bits - len(bit_array)))
        
        return bit_array[:max_bits]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize question + response
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
        
        # Ground truth reasoning steps as bits
        reasoning_bits = []
        for step in sample['reasoning_steps'][:3]:  # Take first 3 steps
            bits = self.text_to_bits(step, max_bits=64)
            reasoning_bits.append(bits)
        
        # Pad if less than 3 steps
        while len(reasoning_bits) < 3:
            reasoning_bits.append(np.zeros(64, dtype=np.uint8))
        
        reasoning_bits = np.stack(reasoning_bits)  # [3, 64]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),  # For language modeling
            'reasoning_bits': torch.from_numpy(reasoning_bits).float(),  # [3, 64]
            'question': sample['question']
        }


# ========================================
# TRAINING LOOP
# ========================================

class SteganographicTrainer:
    """
    Trainer for steganographic model
    """
    
    def __init__(
        self,
        encoder: GPT2WithSteganographicReasoning,
        decoder: SteganographicReasoningDecoder,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        learning_rate: float = 5e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
        # Loss functions
        self.lm_criterion = nn.CrossEntropyLoss()
        self.reasoning_criterion = nn.BCELoss()
        
        # Loss weights
        self.lm_weight = 1.0
        self.reasoning_weight = 0.5
        self.orthogonality_weight = 0.01
    
    def compute_orthogonality_loss(self, reasoning_signals: torch.Tensor) -> torch.Tensor:
        """
        Encourage different channels to be independent
        
        Args:
            reasoning_signals: [batch, num_channels, signal_length]
            
        Returns:
            orthogonality_loss: scalar
        """
        batch_size, num_channels, signal_length = reasoning_signals.shape
        
        ortho_loss = 0.0
        count = 0
        
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                # Compute correlation between channels
                sig_i = reasoning_signals[:, i, :]  # [batch, signal_length]
                sig_j = reasoning_signals[:, j, :]  # [batch, signal_length]
                
                # Normalize
                sig_i_norm = sig_i - sig_i.mean(dim=1, keepdim=True)
                sig_j_norm = sig_j - sig_j.mean(dim=1, keepdim=True)
                
                # Correlation
                correlation = (sig_i_norm * sig_j_norm).sum(dim=1) / (
                    torch.sqrt((sig_i_norm ** 2).sum(dim=1)) * 
                    torch.sqrt((sig_j_norm ** 2).sum(dim=1)) + 1e-8
                )
                
                # Penalize high correlation
                ortho_loss += (correlation ** 2).mean()
                count += 1
        
        return ortho_loss / count if count > 0 else torch.tensor(0.0)
    
    def train_step(self, batch):
        """
        Single training step
        
        Args:
            batch: Dict with input_ids, attention_mask, labels, reasoning_bits
            
        Returns:
            losses: Dict with loss components
        """
        self.encoder.train()
        self.decoder.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        target_reasoning_bits = batch['reasoning_bits'].to(self.device)  # [batch, 3, 64]
        
        # Forward pass through encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            generate_reasoning=True
        )
        
        lm_loss = encoder_outputs['loss']
        reasoning_signals = encoder_outputs['reasoning_signals']  # [batch, 3, 5000]
        
        # Combine signals for decoder
        combined_signals = reasoning_signals.sum(dim=1)  # [batch, 5000]
        
        # Forward pass through decoder
        decoder_outputs = self.decoder(combined_signals)
        decoded_bits = decoder_outputs['decoded_bits']  # [batch, 3, 64]
        
        # Reasoning reconstruction loss
        reasoning_loss = self.reasoning_criterion(decoded_bits, target_reasoning_bits)
        
        # Orthogonality loss (encourage channel independence)
        ortho_loss = self.compute_orthogonality_loss(reasoning_signals)
        
        # Total loss
        total_loss = (
            self.lm_weight * lm_loss +
            self.reasoning_weight * reasoning_loss +
            self.orthogonality_weight * ortho_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'reasoning_loss': reasoning_loss.item(),
            'ortho_loss': ortho_loss.item()
        }
    
    def validate(self):
        """
        Validation loop
        
        Returns:
            avg_losses: Dict with average validation losses
        """
        if self.val_dataloader is None:
            return {}
        
        self.encoder.eval()
        self.decoder.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'lm_loss': 0.0,
            'reasoning_loss': 0.0,
            'ortho_loss': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                target_reasoning_bits = batch['reasoning_bits'].to(self.device)
                
                # Forward
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    generate_reasoning=True
                )
                
                reasoning_signals = encoder_outputs['reasoning_signals']
                combined_signals = reasoning_signals.sum(dim=1)
                
                decoder_outputs = self.decoder(combined_signals)
                decoded_bits = decoder_outputs['decoded_bits']
                
                # Losses
                lm_loss = encoder_outputs['loss']
                reasoning_loss = self.reasoning_criterion(decoded_bits, target_reasoning_bits)
                ortho_loss = self.compute_orthogonality_loss(reasoning_signals)
                
                total_loss = (
                    self.lm_weight * lm_loss +
                    self.reasoning_weight * reasoning_loss +
                    self.orthogonality_weight * ortho_loss
                )
                
                total_losses['total_loss'] += total_loss.item()
                total_losses['lm_loss'] += lm_loss.item()
                total_losses['reasoning_loss'] += reasoning_loss.item()
                total_losses['ortho_loss'] += ortho_loss.item()
                
                num_batches += 1
        
        # Average
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def train(
        self,
        num_epochs: int = 3,
        save_dir: str = 'checkpoints',
        log_interval: int = 10
    ):
        """
        Main training loop
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            log_interval: Log every N steps
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("="*80)
        print("🎓 TRAINING STEGANOGRAPHIC MODEL")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Train batches: {len(self.train_dataloader)}")
        print(f"  Device: {self.device}")
        print(f"  LM weight: {self.lm_weight}")
        print(f"  Reasoning weight: {self.reasoning_weight}")
        print(f"  Orthogonality weight: {self.orthogonality_weight}")
        print("")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}\n")
            
            # Training
            self.encoder.train()
            self.decoder.train()
            
            epoch_losses = {
                'total_loss': 0.0,
                'lm_loss': 0.0,
                'reasoning_loss': 0.0,
                'ortho_loss': 0.0
            }
            
            progress_bar = tqdm(self.train_dataloader, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                losses = self.train_step(batch)
                
                # Accumulate
                for k, v in losses.items():
                    epoch_losses[k] += v
                
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'total_loss': f"{losses['total_loss']:.4f}",
                    'lm_loss': f"{losses['lm_loss']:.4f}",
                    'reasoning_loss': f"{losses['reasoning_loss']:.4f}"
                })
                
                # Log
                if (batch_idx + 1) % log_interval == 0:
                    avg_losses = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                    print(f"\n  Step {global_step} - "
                          f"Total: {avg_losses['total_loss']:.4f}, "
                          f"LM: {avg_losses['lm_loss']:.4f}, "
                          f"Reasoning: {avg_losses['reasoning_loss']:.4f}, "
                          f"Ortho: {avg_losses['ortho_loss']:.4f}")
            
            # Epoch summary
            avg_epoch_losses = {k: v / len(self.train_dataloader) for k, v in epoch_losses.items()}
            
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"  Avg Total Loss: {avg_epoch_losses['total_loss']:.4f}")
            print(f"  Avg LM Loss: {avg_epoch_losses['lm_loss']:.4f}")
            print(f"  Avg Reasoning Loss: {avg_epoch_losses['reasoning_loss']:.4f}")
            print(f"  Avg Ortho Loss: {avg_epoch_losses['ortho_loss']:.4f}")
            
            # Validation
            if self.val_dataloader:
                print(f"\n🔍 Running validation...")
                val_losses = self.validate()
                
                print(f"  Val Total Loss: {val_losses['total_loss']:.4f}")
                print(f"  Val LM Loss: {val_losses['lm_loss']:.4f}")
                print(f"  Val Reasoning Loss: {val_losses['reasoning_loss']:.4f}")
                
                # Save best model
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    
                    checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': best_val_loss,
                    }, checkpoint_path)
                    
                    print(f"\n  ✓ Saved best model to {checkpoint_path}")
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)
            
            print(f"  ✓ Saved checkpoint to {checkpoint_path}")
        
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETE")
        print("="*80)
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}/")


# ========================================
# MAIN TRAINING SCRIPT
# ========================================

def main():
    """
    Main training script
    """
    
    print("="*80)
    print("🚀 STEGANOGRAPHIC MODEL TRAINING")
    print("="*80)
    print("")
    
    # Configuration
    batch_size = 8
    num_epochs = 5
    learning_rate = 5e-5
    num_train_samples = 1000
    num_val_samples = 200
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ReasoningDataset(
        tokenizer=tokenizer,
        num_samples=num_train_samples
    )
    
    val_dataset = ReasoningDataset(
        tokenizer=tokenizer,
        num_samples=num_val_samples
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print("")
    
    # Initialize models
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
        frequencies=[1.0, 2.0, 3.0]
    )
    
    print("✓ Encoder initialized")
    print("✓ Decoder initialized")
    print("")
    
    # Initialize trainer
    trainer = SteganographicTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=learning_rate,
        device=device
    )
    
    # Train
    trainer.train(
        num_epochs=num_epochs,
        save_dir='checkpoints',
        log_interval=10
    )
    
    print("\n✅ Training complete!")
    print("Checkpoints saved to: checkpoints/")
    print("")


if __name__ == "__main__":
    main()
