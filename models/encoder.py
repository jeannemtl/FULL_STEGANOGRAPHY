import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
import numpy as np

class SteganographicReasoningLayer(nn.Module):
    """Layer that encodes reasoning onto frequency channels"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_channels: int = 3,
        frequencies: list = [1.0, 2.0, 3.0],
        signal_length: int = 5000,
        sampling_rate: int = 100
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.frequencies = frequencies
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        
        # Learnable projectors: hidden state → reasoning bits
        self.reasoning_projector = nn.ModuleList([
            nn.Linear(hidden_size, 64) for _ in range(num_channels)
        ])
        
        # Modulation parameters
        self.amplitude_high = nn.Parameter(torch.tensor(1.0))
        self.amplitude_low = nn.Parameter(torch.tensor(0.1))
        
        # Time vector for carrier generation
        self.register_buffer(
            'time',
            torch.linspace(0, signal_length/sampling_rate, signal_length)
        )
    
    def encode_reasoning_to_signal(
        self,
        reasoning_bits: torch.Tensor,
        frequency: float
    ) -> torch.Tensor:
        """Encode bits onto carrier using ASK modulation"""
        
        batch_size = reasoning_bits.shape[0]
        samples_per_bit = self.signal_length // 64
        
        # Carrier wave
        carrier = torch.sin(2 * np.pi * frequency * self.time)
        carrier = carrier.unsqueeze(0).expand(batch_size, -1)
        
        # Expand bits
        bits_expanded = reasoning_bits.unsqueeze(2).repeat(1, 1, samples_per_bit)
        bits_expanded = bits_expanded.reshape(batch_size, -1)
        
        # ASK modulation
        amplitude = torch.where(
            bits_expanded > 0.5,
            self.amplitude_high,
            self.amplitude_low
        )
        
        signal = amplitude * carrier
        return signal
    
    def forward(self, hidden_states: torch.Tensor, generate_signal: bool = False):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            generate_signal: Whether to generate FDM signals
        
        Returns:
            reasoning_signals or reasoning_logits
        """
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        
        reasoning_logits = []
        reasoning_signals = []
        
        for i, (projector, freq) in enumerate(zip(self.reasoning_projector, self.frequencies)):
            logits = projector(last_hidden)  # [batch, 64]
            reasoning_logits.append(logits)
            
            if generate_signal:
                bits = torch.sigmoid(logits)
                signal = self.encode_reasoning_to_signal(bits, freq)
                reasoning_signals.append(signal)
        
        reasoning_logits = torch.stack(reasoning_logits, dim=1)  # [batch, num_channels, 64]
        
        if generate_signal:
            reasoning_signals = torch.stack(reasoning_signals, dim=1)  # [batch, num_channels, signal_length]
            return reasoning_signals, reasoning_logits
        
        return reasoning_logits


class GPT2WithSteganographicReasoning(nn.Module):
    """GPT-2 with built-in steganographic reasoning"""
    
    def __init__(
        self,
        base_model_name: str = 'gpt2',
        num_channels: int = 3,
        frequencies: list = [1.0, 2.0, 3.0]
    ):
        super().__init__()
        
        self.gpt2 = GPT2LMHeadModel.from_pretrained(base_model_name)
        config = self.gpt2.config
        
        self.stego_layer = SteganographicReasoningLayer(
            hidden_size=config.n_embd,
            num_channels=num_channels,
            frequencies=frequencies
        )
        
        self.modulation_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        generate_reasoning: bool = False
    ):
        """Forward pass"""
        
        gpt2_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = gpt2_outputs.last_hidden_state
        lm_logits = self.gpt2.lm_head(hidden_states)
        
        if generate_reasoning:
            reasoning_signals, reasoning_logits = self.stego_layer(
                hidden_states, generate_signal=True
            )
        else:
            reasoning_logits = self.stego_layer(hidden_states, generate_signal=False)
            reasoning_signals = None
        
        modulated_logits = lm_logits + self.modulation_weight * reasoning_logits.sum(dim=1).unsqueeze(1)
        
        loss = None
        if labels is not None:
            shift_logits = modulated_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'logits': modulated_logits,
            'reasoning_signals': reasoning_signals,
            'reasoning_logits': reasoning_logits,
            'loss': loss
        }
    
    def generate_with_reasoning(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0
    ):
        """Generate text with embedded reasoning"""
        
        device = input_ids.device
        
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.forward(input_ids=input_ids, generate_reasoning=False)
            
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        final_outputs = self.forward(input_ids=input_ids, generate_reasoning=True)
        
        return input_ids, final_outputs['reasoning_signals']