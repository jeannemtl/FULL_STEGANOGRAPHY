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
from training.dataset import ReasoningDataset


class SteganographicTrainer:
    """Trainer for steganographic model"""
    
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
        
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
        self.lm_criterion = nn.CrossEntropyLoss()
        self.reasoning_criterion = nn.BCELoss()
        
        self.lm_weight = 1.0
        self.reasoning_weight = 0.5
        self.orthogonality_weight = 0.01
    
    def compute_orthogonality_loss(self, reasoning_signals: torch.Tensor) -> torch.Tensor:
        """Encourage channel independence"""
        
        batch_size, num_channels, signal_length = reasoning_signals.shape
        
        ortho_loss = 0.0
        count = 0
        
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                sig_i = reasoning_signals[:, i, :]
                sig_j = reasoning_signals[:, j, :]
                
                sig_i_norm = sig_i - sig_i.mean(dim=1, keepdim=True)
                sig_j_norm = sig_j - sig_j.mean(dim=1, keepdim=True)
                
                correlation = (sig_i_norm * sig_j_norm).sum(dim=1) / (
                    torch.sqrt((sig_i_norm ** 2).sum(dim=1)) * 
                    torch.sqrt((sig_j_norm ** 2).sum(dim=1)) + 1e-8
                )
                
                ortho_loss += (correlation ** 2).mean()
                count += 1
        
        return ortho_loss / count if count > 0 else torch.tensor(0.0)
    
    def train_step(self, batch):
        """Single training step"""
        
        self.encoder.train()
        self.decoder.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        target_reasoning_bits = batch['reasoning_bits'].to(self.device)
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            generate_reasoning=True
        )
        
        lm_loss = encoder_outputs['loss']
        reasoning_signals = encoder_outputs['reasoning_signals']
        
        combined_signals = reasoning_signals.sum(dim=1)
        
        decoder_outputs = self.decoder(combined_signals)
        decoded_bits = decoder_outputs['decoded_bits']
        
        reasoning_loss = self.reasoning_criterion(decoded_bits, target_reasoning_bits)
        ortho_loss = self.compute_orthogonality_loss(reasoning_signals)
        
        total_loss = (
            self.lm_weight * lm_loss +
            self.reasoning_weight * reasoning_loss +
            self.orthogonality_weight * ortho_loss
        )
        
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
        """Validation loop"""
        
        if self.val_dataloader is None:
            return {}
        
        self.encoder.eval()
        self.decoder.eval()
        
        total_losses = {'total_loss': 0.0, 'lm_loss': 0.0, 'reasoning_loss': 0.0, 'ortho_loss': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                target_reasoning_bits = batch['reasoning_bits'].to(self.device)
                
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
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def train(self, num_epochs: int = 3, save_dir: str = 'checkpoints', log_interval: int = 10):
        """Main training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("="*80)
        print("🎓 TRAINING STEGANOGRAPHIC MODEL")
        print("="*80)
        print(f"\nEpochs: {num_epochs}")
        print(f"Train batches: {len(self.train_dataloader)}")
        print(f"Device: {self.device}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}\n")
            
            self.encoder.train()
            self.decoder.train()
            
            epoch_losses = {'total_loss': 0.0, 'lm_loss': 0.0, 'reasoning_loss': 0.0, 'ortho_loss': 0.0}
            
            progress_bar = tqdm(self.train_dataloader, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                losses = self.train_step(batch)
                
                for k, v in losses.items():
                    epoch_losses[k] += v
                
                progress_bar.set_postfix({
                    'total': f"{losses['total_loss']:.4f}",
                    'lm': f"{losses['lm_loss']:.4f}",
                    'reasoning': f"{losses['reasoning_loss']:.4f}"
                })
            
            avg_epoch_losses = {k: v / len(self.train_dataloader) for k, v in epoch_losses.items()}
            
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"  Avg Total: {avg_epoch_losses['total_loss']:.4f}")
            print(f"  Avg LM: {avg_epoch_losses['lm_loss']:.4f}")
            print(f"  Avg Reasoning: {avg_epoch_losses['reasoning_loss']:.4f}")
            
            if self.val_dataloader:
                val_losses = self.validate()
                print(f"\n🔍 Validation:")
                print(f"  Val Total: {val_losses['total_loss']:.4f}")
                
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
                    print(f"  ✓ Saved best model")
            
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)
        
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETE")
        print("="*80)


def main():
    """Main training script"""
    
    batch_size = 8
    num_epochs = 5
    learning_rate = 5e-5
    num_train_samples = 1000
    num_val_samples = 200
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = ReasoningDataset(tokenizer=tokenizer, num_samples=num_train_samples)
    val_dataset = ReasoningDataset(tokenizer=tokenizer, num_samples=num_val_samples)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    encoder = GPT2WithSteganographicReasoning()
    decoder = SteganographicReasoningDecoder()
    
    trainer = SteganographicTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=learning_rate,
        device=device
    )
    
    trainer.train(num_epochs=num_epochs, save_dir='checkpoints')


if __name__ == "__main__":
    main()