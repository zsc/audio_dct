import sys
import numpy as np
import librosa
import math

def calculate_psnr(file1, file2, target_sr=16000):
    try:
        # Load and automatically resample to target_sr
        # librosa loads as float32, normalized to [-1, 1]
        y1, sr1 = librosa.load(file1, sr=target_sr, mono=True)
        y2, sr2 = librosa.load(file2, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    # Ensure same length
    min_len = min(len(y1), len(y2))
    if len(y1) != len(y2):
        print(f"Warning: Lengths differ ({len(y1)} vs {len(y2)}). Truncating to {min_len} samples.")
    
    d1 = y1[:min_len]
    d2 = y2[:min_len]
    
    # Since librosa normalizes to [-1, 1], MAX is 2.0 (peak-to-peak) or 1.0 (amplitude)?
    # Standard PSNR for signals usually uses the dynamic range. 
    # For [-1, 1] float audio, the dynamic range is 2. 
    # However, often people use MAX=1 if considering peak amplitude, or MAX=2 if peak-to-peak.
    # But strictly speaking, if the signal is bounded by [-1, 1], the maximum possible error power is based on that range.
    # In image processing (0-255), MAX is 255.
    # For audio, standard convention varies. 
    # Often for normalized audio, MAX=1 is used (assuming silence vs full scale).
    # Let's use MAX=1.0 consistent with standard definition R = 1.
    
    max_val = 1.0

    mse = np.mean((d1 - d2) ** 2)
    
    if mse == 0:
        print("PSNR: Infinity (Signals are identical)")
        return

    psnr = 10 * np.log10((max_val ** 2) / mse)
    
    print(f"MSE: {mse:.6f}")
    print(f"MAX: {max_val}")
    print(f"PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_psnr.py <file1.wav> <file2.wav>")
        sys.exit(1)
    
    calculate_psnr(sys.argv[1], sys.argv[2])
