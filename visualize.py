import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import argparse
from st_dct import st_dct, get_mel_basis, mu_law_encode

def generate_visualizations(audio_file, height=192, use_mel=True):
    print(f"Generating visualizations for {audio_file}...")
    
    if not os.path.exists(audio_file):
        print(f"File {audio_file} not found.")
        return

    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    
    # 1. Waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title("Input Waveform")
    plt.tight_layout()
    plt.savefig("input_vis.png", dpi=120)
    plt.close()
    print("Saved input_vis.png")

    # 2. ST-DCT Spectrogram
    if use_mel:
        n_dct = 1024
        # hop_length = n_dct // 4
    else:
        n_dct = height
        
    hop_length = n_dct // 4
    
    # Use st_dct from the module
    # Pad first
    pad_length = n_dct // 2
    y_padded = np.pad(y, pad_length, mode='reflect')
    
    spec = st_dct(y_padded, n_fft=n_dct, hop_length=hop_length)
    
    # Convert to dB for visualization (or use the log-compressed / mu-law form)
    # Let's show the mu-law compressed form as that's what goes into the image
    
    pos_spec = np.maximum(0, spec)
    neg_spec = np.maximum(0, -spec)
    
    if use_mel:
        mel_basis = get_mel_basis(sr, n_dct, height)
        pos_spec = np.dot(mel_basis, pos_spec)
        neg_spec = np.dot(mel_basis, neg_spec)
        
    # Combine for visualization: just magnitude or difference?
    # Let's show magnitude (sum) and sign (color?) or just the encoded channels.
    # To keep it simple and like a spectrogram:
    vis_spec = pos_spec - neg_spec # Reconstruct signed
    
    # Plot standard spectrogram-like view (dB)
    plt.figure(figsize=(10, 4))
    # We can use librosa.display.specshow, but we need to adapt coordinates
    # For DCT, the y-axis is coefficient index / frequency
    
    # Log scale for visibility
    vis_img = librosa.amplitude_to_db(np.abs(vis_spec), ref=np.max)
    
    librosa.display.specshow(vis_img, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel' if use_mel else 'linear', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"ST-DCT Spectrogram ({'Mel' if use_mel else 'Linear'})")
    plt.tight_layout()
    plt.savefig("spectrogram.png", dpi=120)
    plt.close()
    print("Saved spectrogram.png")
    
    # 3. Encoded Channels Visualization (Red/Blue)
    # Show what actually goes into the image
    max_val = max(np.max(pos_spec), np.max(neg_spec))
    if max_val == 0: max_val = 1
    
    pos_norm = pos_spec / max_val
    neg_norm = neg_spec / max_val
    
    # Apply mu-law
    pos_norm = mu_law_encode(pos_norm)
    neg_norm = mu_law_encode(neg_norm)
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.imshow(pos_norm, aspect='auto', origin='lower', cmap='Reds')
    plt.title("Positive Channel (Red)")
    plt.colorbar()
    
    plt.subplot(2, 1, 2)
    plt.imshow(neg_norm, aspect='auto', origin='lower', cmap='Blues')
    plt.title("Negative Channel (Blue)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("dct_vis.png", dpi=120)
    plt.close()
    print("Saved dct_vis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?', default="input.wav")
    args = parser.parse_args()
    
    generate_visualizations(args.input_file)
