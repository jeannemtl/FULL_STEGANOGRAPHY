import numpy as np

def calculate_accuracy(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    """Calculate bit accuracy percentage"""
    correct = np.sum(original_bits == recovered_bits)
    total = len(original_bits)
    return (correct / total) * 100.0

def calculate_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((original - reconstructed) ** 2))