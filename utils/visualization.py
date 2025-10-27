import matplotlib.pyplot as plt
import numpy as np

def plot_fft_spectrum(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    carrier_freqs: list,
    agent_names: list = None,
    save_path: str = None,
    title: str = "FFT Spectrum"
):
    """Plot FFT spectrum with carrier frequencies marked"""
    
    if agent_names is None:
        agent_names = [f"Channel {i+1}" for i in range(len(carrier_freqs))]
    
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, magnitudes, 'b-', linewidth=1.5, label='Signal')
    
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    
    for i, (freq, name) in enumerate(zip(carrier_freqs, agent_names)):
        color = colors[i % len(colors)]
        plt.axvline(x=freq, color=color, linestyle='--', linewidth=2, 
                   label=f'{name} ({freq} Hz)')
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(carrier_freqs) + 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()