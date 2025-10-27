import numpy as np

class OTPEncryption:
    """One-Time Pad encryption for perfect secrecy"""
    
    def quantize_signal(
        self,
        signal: np.ndarray,
        bits_per_sample: int = 8
    ) -> tuple:
        """Quantize continuous signal to bits"""
        s_min, s_max = signal.min(), signal.max()
        
        # Normalize to [0, 2^bits - 1]
        normalized = (signal - s_min) / (s_max - s_min + 1e-10)
        quantized = (normalized * (2**bits_per_sample - 1)).astype(np.uint8)
        
        # Convert to bits
        bits = np.unpackbits(quantized)
        
        return bits, s_min, s_max
    
    def dequantize_signal(
        self,
        bits: np.ndarray,
        s_min: float,
        s_max: float,
        bits_per_sample: int = 8
    ) -> np.ndarray:
        """Reconstruct signal from bits"""
        # Pack bits to bytes
        quantized = np.packbits(bits)
        
        # Denormalize
        normalized = quantized.astype(np.float64) / (2**bits_per_sample - 1)
        signal = normalized * (s_max - s_min) + s_min
        
        return signal
    
    def encrypt(self, plaintext_bits: np.ndarray) -> tuple:
        """Encrypt with one-time pad"""
        key = np.random.randint(0, 2, size=len(plaintext_bits), dtype=np.uint8)
        ciphertext = np.bitwise_xor(plaintext_bits, key)
        return ciphertext, key
    
    def decrypt(self, ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Decrypt with one-time pad"""
        plaintext = np.bitwise_xor(ciphertext, key)
        return plaintext