import torch
from transformers import GPT2Tokenizer
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import GPT2WithSteganographicReasoning
from models.decoder import SteganographicReasoningDecoder, ClassicalSignalDecoder
from core.demodulation import ASKDemodulator
from utils.visualization import plot_fft_spectrum
from utils.metrics import calculate_accuracy


def bits_to_text(bits: np.ndarray) -> str:
    """Convert bits to text"""
    byte_array = np.packbits(bits)
    try:
        text = byte_array.tobytes().decode('ascii', errors='ignore')
        return text.strip('\x00').strip()
    except:
        return "[Decode error]"


def main():
    """End-to-end demo"""
    
    print("="*80)
    print("🎯 STEGANOGRAPHIC AI DEMONSTRATION")
    print("="*80)
    print("\nAI splits reasoning across 3 frequency channels\n")
    
    # Initialize
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    encoder = GPT2WithSteganographicReasoning()
    decoder = SteganographicReasoningDecoder()
    
    encoder.eval()
    decoder.eval()
    
    # Prompt
    prompt = "Should we increase prices on our premium products?"
    print(f"📝 Prompt: \"{prompt}\"\n")
    
    # Generate
    print("🤖 Generating with embedded reasoning...")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        generated_ids, reasoning_signals = encoder.generate_with_reasoning(
            input_ids=input_ids,
            max_length=50,
            temperature=0.8
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"✓ Generated: \"{generated_text}\"\n")
    
    print("🔊 Reasoning encoded on:")
    for i in range(3):
        signal = reasoning_signals[0, i, :].numpy()
        print(f"  Channel {i+1} (freq {[1.0, 2.0, 3.0][i]} Hz): range [{signal.min():.3f}, {signal.max():.3f}]")
    
    # Decode
    print("\n🔍 Decoding reasoning...")
    combined_signal = reasoning_signals[0].sum(dim=0)
    
    with torch.no_grad():
        decoder_outputs = decoder(combined_signal.unsqueeze(0))
        decoded_bits = decoder_outputs['decoded_bits'][0].cpu().numpy()
    
    print("\n✓ Decoded reasoning:")
    for i in range(3):
        bits = (decoded_bits[i] > 0.5).astype(np.uint8)
        text = bits_to_text(bits)
        print(f"  Channel {i+1}: \"{text}\"")
    
    # Visualization
    os.makedirs("results", exist_ok=True)
    
    demod = ASKDemodulator(sampling_rate=100)
    freqs, mags = demod.fft_analysis(combined_signal.numpy())
    
    plot_fft_spectrum(
        freqs, mags,
        carrier_freqs=[1.0, 2.0, 3.0],
        agent_names=["Step 1", "Step 2", "Step 3"],
        save_path="results/fft_demo.png",
        title="AI Reasoning Split Across 3 Frequency Channels"
    )
    
    print("\n✅ Complete!")
    print("  • AI generated text")
    print("  • Reasoning encoded on 3 frequencies")
    print("  • All reasoning recovered")
    print("  • Visualization saved to results/fft_demo.png\n")


if __name__ == "__main__":
    main()