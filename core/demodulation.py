import numpy as np
from scipy import signal

class ASKDemodulator:
    """ASK signal demodulator"""
    
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
    
    def bandpass_filter(
        self,
        signal_data: np.ndarray,
        center_freq: float,
        bandwidth: float = 0.4
    ) -> np.ndarray:
        """Apply bandpass filter"""
        low = center_freq - bandwidth
        high = center_freq + bandwidth
        
        sos = signal.butter(4, [low, high], btype='band', fs=self.sampling_rate, output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        
        return filtered
    
    def extract_envelope(self, signal_data: np.ndarray) -> np.ndarray:
        """Extract envelope using Hilbert transform"""
        analytic = signal.hilbert(signal_data)
        envelope = np.abs(analytic)
        return envelope
    
    def demodulate(
        self,
        signal_data: np.ndarray,
        carrier_freq: float,
        num_bits: int = 64
    ) -> np.ndarray:
        """Demodulate ASK signal"""
        # Filter
        filtered = self.bandpass_filter(signal_data, carrier_freq)
        
        # Extract envelope
        envelope = self.extract_envelope(filtered)
        
        # Decode bits
        samples_per_bit = len(envelope) // num_bits
        bits = np.zeros(num_bits, dtype=np.uint8)
        
        threshold = (envelope.max() + envelope.min()) / 2
        
        for i in range(num_bits):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            avg = np.mean(envelope[start:end])
            bits[i] = 1 if avg > threshold else 0
        
        return bits
    
    def fft_analysis(self, signal_data: np.ndarray) -> tuple:
        """FFT analysis"""
        n = len(signal_data)
        fft_result = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(n, 1/self.sampling_rate)
        
        # Positive frequencies only
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        magnitudes = np.abs(fft_result[positive_mask])
        
        return freqs, magnitudes