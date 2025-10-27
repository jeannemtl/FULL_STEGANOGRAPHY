import numpy as np

class FDMSignal:
    """Frequency Division Multiplexing signal generator"""
    
    def __init__(self, num_samples: int = 5000, sampling_rate: int = 100):
        self.num_samples = num_samples
        self.sampling_rate = sampling_rate
        self.time = np.linspace(0, num_samples/sampling_rate, num_samples)
    
    def create_ask_signal(self, bits: np.ndarray, carrier_freq: float) -> np.ndarray:
        """
        Create ASK modulated signal
        
        Args:
            bits: Binary array (0s and 1s)
            carrier_freq: Carrier frequency in Hz
            
        Returns:
            signal: ASK modulated signal
        """
        num_bits = len(bits)
        samples_per_bit = self.num_samples // num_bits
        
        # Create carrier wave
        carrier = np.sin(2 * np.pi * carrier_freq * self.time)
        
        # Amplitude modulation
        signal = np.zeros(self.num_samples)
        
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = min((i + 1) * samples_per_bit, self.num_samples)
            
            amplitude = 1.0 if bit == 1 else 0.1
            signal[start:end] = amplitude * carrier[start:end]
        
        return signal
    
    def combine_signals(self, signals: list) -> np.ndarray:
        """Combine multiple signals (FDM)"""
        combined = np.sum(signals, axis=0)
        return combined
    
    def get_cycles_per_bit(self, frequency: float, num_bits: int) -> float:
        """Calculate cycles per bit"""
        samples_per_bit = self.num_samples / num_bits
        duration_per_bit = samples_per_bit / self.sampling_rate
        return frequency * duration_per_bit