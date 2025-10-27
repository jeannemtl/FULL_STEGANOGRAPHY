import torch
import torch.nn as nn
import numpy as np
from scipy import signal as scipy_signal

class SteganographicReasoningDecoder(nn.Module):
    """Neural decoder for extracting reasoning from signals"""
    
    def __init__(
        self,
        signal_length: int = 5000,
        sampling_rate: int = 100,
        num_channels: int = 3,
        frequencies: list = [1.0, 2.0, 3.0],
        hidden_size: int = 256
    ):
        super().__init__()
        
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.frequencies = frequencies
        
        self.filter_bandwidth = nn.Parameter(torch.tensor(0.4))
        
        # Envelope detectors
        self.envelope_detector = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=31, padding=15),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=15, padding=7),
                nn.ReLU(),
                nn.Conv1d(64, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )
            for _ in range(num_channels)
        ])
        
        # Bit decoders
        self.bit_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(signal_length, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 64),
                nn.Sigmoid()
            )
            for _ in range(num_channels)
        ])
    
    def bandpass_filter(self, signal_data: torch.Tensor, center_freq: float) -> torch.Tensor:
        """Apply bandpass filter"""
        
        signal_np = signal_data.detach().cpu().numpy()
        batch_size = signal_data.shape[0]
        
        low = center_freq - self.filter_bandwidth.item()
        high = center_freq + self.filter_bandwidth.item()
        
        sos = scipy_signal.butter(4, [low, high], btype='band', fs=self.sampling_rate, output='sos')
        
        filtered_np = np.zeros_like(signal_np)
        for i in range(batch_size):
            filtered_np[i] = scipy_signal.sosfilt(sos, signal_np[i])
        
        return torch.from_numpy(filtered_np).to(signal_data.device).float()
    
    def decode_channel(self, signal_data: torch.Tensor, channel_idx: int) -> torch.Tensor:
        """Decode one channel"""
        
        freq = self.frequencies[channel_idx]
        
        # Bandpass filter
        filtered = self.bandpass_filter(signal_data, freq)
        
        # Extract envelope
        signal_3d = filtered.unsqueeze(1)
        envelope = self.envelope_detector[channel_idx](signal_3d).squeeze(1)
        
        # Decode bits
        bits = self.bit_decoder[channel_idx](envelope)
        
        return bits
    
    def forward(self, signal_data: torch.Tensor):
        """Decode all channels"""
        
        all_bits = []
        for channel_idx in range(self.num_channels):
            bits = self.decode_channel(signal_data, channel_idx)
            all_bits.append(bits)
        
        decoded_bits = torch.stack(all_bits, dim=1)  # [batch, num_channels, 64]
        
        return {'decoded_bits': decoded_bits}


class ClassicalSignalDecoder:
    """Classical signal processing decoder"""
    
    def __init__(self, signal_length: int = 5000, sampling_rate: int = 100, frequencies: list = [1.0, 2.0, 3.0]):
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        self.frequencies = frequencies
    
    def bandpass_filter(self, signal_data: np.ndarray, center_freq: float, bandwidth: float = 0.4) -> np.ndarray:
        """Butterworth bandpass filter"""
        
        low = center_freq - bandwidth
        high = center_freq + bandwidth
        sos = scipy_signal.butter(4, [low, high], btype='band', fs=self.sampling_rate, output='sos')
        return scipy_signal.sosfilt(sos, signal_data)
    
    def extract_envelope(self, signal_data: np.ndarray) -> np.ndarray:
        """Hilbert transform envelope"""
        
        analytic = scipy_signal.hilbert(signal_data)
        return np.abs(analytic)
    
    def decode_bits(self, envelope: np.ndarray, num_bits: int = 64) -> np.ndarray:
        """Decode bits from envelope"""
        
        samples_per_bit = len(envelope) // num_bits
        bits = np.zeros(num_bits, dtype=np.uint8)
        threshold = (envelope.max() + envelope.min()) / 2
        
        for i in range(num_bits):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            bits[i] = 1 if np.mean(envelope[start:end]) > threshold else 0
        
        return bits
    
    def decode_channel(self, signal_data: np.ndarray, frequency: float) -> np.ndarray:
        """Decode one channel"""
        
        filtered = self.bandpass_filter(signal_data, frequency)
        envelope = self.extract_envelope(filtered)
        bits = self.decode_bits(envelope, num_bits=64)
        return bits
    
    def decode_all_channels(self, signal_data: np.ndarray) -> dict:
        """Decode all channels"""
        
        decoded_bits_all = [self.decode_channel(signal_data, freq) for freq in self.frequencies]
        return {'decoded_bits': np.stack(decoded_bits_all, axis=0)}