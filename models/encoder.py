"""
GPT-2 Encoder with Steganographic Reasoning Layer
Encodes reasoning across multiple frequency channels using FDM
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import numpy as np


class SteganographicReasoningLayer(nn.Module):
    """
    Layer that encodes reasoning bits into frequency-domain signals
    """
    
    def __init__(
        self,
        signal_length: int = 5000,
        sampling_rate: int = 100,
        num_channels: int = 3,
        frequencies: list = None
    ):
        super().__init__()
        
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.frequencies = frequencies or [1.0, 2.0, 3.0]
        
        # Learnable projection for reasoning bits
        self.reasoning_projector = nn.Linear(768, num_channels * 64)
    
    def encode_reasoning_to_signal(self, bits: torch.Tensor, carrier_freq: float) -> torch.Tensor:
        """
        Encode reasoning bits into a signal at the specified carrier frequency
        
        Args:
            bits: [batch_size, num_bits] binary tensor
            carrier_freq: carrier frequency in Hz
            
        Returns:
            signal: [batch_size, signal_length] modulated signal
        """
        batch_size = bits.shape[0]
        num_bits = bits.shape[1]
        
        # Calculate samples per bit (use integer division)
        samples_per_bit = self.signal_length // num_bits
        
        # Expand bits to signal
        expanded_bits = bits.unsqueeze(-1).repeat(1, 1, samples_per_bit)
        expanded_bits = expanded_bits.reshape(batch_size, -1)
        
        # PAD to exact signal_length
        current_length = expanded_bits.shape[1]
        if current_length < self.signal_length:
            padding = self.signal_length - current_length
            expanded_bits = torch.nn.functional.pad(expanded_bits, (0, padding), value=0)
        elif current_length > self.signal_length:
            expanded_bits = expanded_bits[:, :self.signal_length]
        
        # Generate time array
        t = torch.linspace(0, self.signal_length / self.sampling_rate, 
                           self.signal_length, device=bits.device)
        t = t.unsqueeze(0).expand(batch_size, -1)
        
        # Generate carrier
        carrier = torch.sin(2 * np.pi * carrier_freq * t)
        
        # ASK modulation
        amplitude = expanded_bits * 0.5 + 0.5  # Scale to [0.5, 1.0]
        signal = amplitude * carrier
        
        return signal
    
    def forward(self, hidden_states: torch.Tensor, reasoning_bits: torch.Tensor = None):
        """
        Forward pass: encode reasoning into frequency signals
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] from GPT-2
            reasoning_bits: [batch_size, num_channels, num_bits] optional target bits
            
        Returns:
            reasoning_signals: [batch_size, num_channels, signal_length]
            reasoning_logits: [batch_size, num_channels, num_bits] for training
        """
        batch_size = hidden_states.shape[0]
        
        # Use last hidden state as reasoning representation
        reasoning_repr = hidden_states[:, -1, :]  # [batch_size, 768]
        
        # Project to bit logits
        reasoning_logits = self.reasoning_projector(reasoning_repr)
        reasoning_logits = reasoning_logits.view(batch_size, self.num_channels, 64)
        
        # Convert to bits (sigmoid for soft bits during training)
        reasoning_bits_pred = torch.sigmoid(reasoning_logits)
        
        # Use provided bits if available, otherwise use predicted
        if reasoning_bits is not None:
            bits_to_encode = reasoning_bits
        else:
            bits_to_encode = reasoning_bits_pred
        
        # Encode each channel at its frequency
        reasoning_signals = []
        for i in range(self.num_channels):
            signal = self.encode_reasoning_to_signal(
                bits_to_encode[:, i, :],
                self.frequencies[i]
            )
            reasoning_signals.append(signal)
        
        reasoning_signals = torch.stack(reasoning_signals, dim=1)
        
        return reasoning_signals, reasoning_logits


class GPT2WithSteganographicReasoning(nn.Module):
    """
    GPT-2 model with steganographic reasoning encoding
    """
    
    def __init__(
        self,
        base_model_name: str = 'gpt2',
        num_channels: int = 3,
        frequencies: list = None
    ):
        super().__init__()
        
        # Load base GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(base_model_name)
        
        # Add steganographic layer
        self.stego_layer = SteganographicReasoningLayer(
            signal_length=5000,
            sampling_rate=100,
            num_channels=num_channels,
            frequencies=frequencies or [1.0, 2.0, 3.0]
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        reasoning_bits: torch.Tensor = None,
        generate_reasoning: bool = True
    ):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] for language modeling loss
            reasoning_bits: [batch_size, num_channels, num_bits] target reasoning
            generate_reasoning: whether to generate reasoning signals
            
        Returns:
            dict with lm_loss, reasoning_signals, reasoning_logits, logits
        """
        
        # Forward through GPT-2
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        lm_loss = outputs.loss
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Generate reasoning signals
        reasoning_signals = None
        reasoning_logits = None
        
        if generate_reasoning:
            reasoning_signals, reasoning_logits = self.stego_layer(
                hidden_states,
                reasoning_bits
            )
        
        return {
            'lm_loss': lm_loss,
            'logits': logits,
            'reasoning_signals': reasoning_signals,
            'reasoning_logits': reasoning_logits
        }
    
    def generate_with_reasoning(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        **kwargs
    ):
        """
        Generate text with reasoning signals
        
        Args:
            input_ids: [batch_size, seq_len]
            max_length: maximum generation length
            temperature: sampling temperature
            
        Returns:
            generated_ids: [batch_size, max_length]
            reasoning_signals: [batch_size, num_channels, signal_length]
        """
        
        # Generate text
        generated_ids = self.gpt2.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.gpt2.config.eos_token_id,
            **kwargs
        )
        
        # Get final hidden states for reasoning
        with torch.no_grad():
            outputs = self.gpt2(
                input_ids=generated_ids,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            
            # Generate reasoning signals
            reasoning_signals, _ = self.stego_layer(hidden_states, reasoning_bits=None)
        
        return generated_ids, reasoning_signals
