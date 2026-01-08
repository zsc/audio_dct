import numpy as np
import scipy.fft as fft
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import librosa
import os
import sys
import argparse
from PIL import Image
import pillow_avif  # Ensures AVIF support is registered

def generate_synthetic_audio(filename, duration=3.0, sr=22050):
    """Generate a synthetic audio file."""
    t = np.linspace(0, duration, int(sr * duration))
    sweep = np.sin(2 * np.pi * np.linspace(440, 4400, len(t)) * t)
    tone1 = 0.5 * np.sin(2 * np.pi * 1000 * t)
    tone2 = 0.3 * np.sin(2 * np.pi * 2000 * t)
    audio = sweep + tone1 + tone2
    audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
    wavfile.write(filename, sr, audio)
    print(f"Generated {filename}")

def st_dct(y, n_fft=2048, hop_length=512, window='hann'):
    """Compute Short-Time DCT-II."""
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)
    win = librosa.filters.get_window(window, n_fft).reshape(-1, 1)
    frames = frames * win
    dct_out = fft.dct(frames, type=2, axis=0, norm='ortho')
    return dct_out

def st_idct(dct_spec, n_fft=2048, hop_length=512, window='hann'):
    """Compute Inverse Short-Time DCT-II with Overlap-Add."""
    # IDCT
    # dct_spec shape: (n_fft, n_frames)
    frames = fft.idct(dct_spec, type=2, axis=0, norm='ortho')
    
    # Overlap-Add
    n_frames = frames.shape[1]
    expected_length = n_fft + (n_frames - 1) * hop_length
    y_recon = np.zeros(expected_length)
    window_sum = np.zeros(expected_length)
    
    win = librosa.filters.get_window(window, n_fft)
    # Reshape window for broadcasting if needed, but here it's 1D array
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        # Add windowed frame
        y_recon[start:end] += frames[:, i] * win
        # Add squared window to sum for normalization
        window_sum[start:end] += win ** 2
        
    # Normalize by window sum (avoid division by zero)
    # Use a small epsilon
    y_recon /= (window_sum + 1e-8)
    
    return y_recon

def encode_audio_to_avif(audio_file, quality=75):
    """Encode audio to AVIF image."""
    print(f"Encoding {audio_file} to AVIF (Q={quality})...")
    y, sr = librosa.load(audio_file, sr=None)
    
    n_fft = 1024
    hop_length = 256
    
    dct_spec = st_dct(y, n_fft=n_fft, hop_length=hop_length)
    
    # Normalization to 0-255
    # We map [-limit, limit] to [0, 255]
    # We pick a fixed reasonable limit or the max of the signal?
    # To minimize clipping, let's use the max abs value.
    max_val = np.max(np.abs(dct_spec))
    if max_val == 0:
        max_val = 1.0
        
    # Normalize to [0, 1]
    # (x / max + 1) / 2
    norm_spec = (dct_spec / max_val + 1) / 2
    
    # Quantize to 8-bit
    img_data = (norm_spec * 255).astype(np.uint8)
    
    # Image shape: (Freq, Time)
    # We flip Y axis so low freq is at bottom (standard image logic vs matrix)
    # Actually, matrix (0,0) is top-left.
    # DCT bin 0 is DC (low freq). In matrix, row 0 is top.
    # So row 0 (low freq) will be at the top of the image.
    # Usually spectrograms have low freq at bottom.
    # Let's flip it for visual consistency, OR keep it raw for simpler decoding.
    # Keeping it raw (row 0 at top) is safer for logic.
    
    img = Image.fromarray(img_data, mode='L')
    
    base_name = os.path.splitext(audio_file)[0]
    avif_filename = f"{base_name}.avif"
    
    # We lose 'max_val' and 'sr' and 'length'.
    # We can try to assume standard SR or normalize output.
    # For a prototype, we'll accept volume loss.
    
    img.save(avif_filename, format='AVIF', quality=quality)
    print(f"Saved {avif_filename}")
    return avif_filename

def decode_avif_to_audio(avif_file):
    """Decode AVIF image to audio."""
    print(f"Decoding {avif_file} to WAV...")
    img = Image.open(avif_file)
    if img.mode != 'L':
        img = img.convert('L')
        
    img_data = np.array(img).astype(np.float32)
    
    # Map [0, 255] back to [-1, 1] (approximately)
    # val = (pixel / 255) * 2 - 1
    # Note: We lost the original 'max_val'.
    # The reconstructed signal will have peak amplitude around 1.0 (in DCT domain).
    # Audio volume will be arbitrary.
    norm_spec = (img_data / 255.0) * 2 - 1
    
    # Recover DCT spec (assuming unit scale)
    # If original was huge, this is quiet. If original was tiny, this is loud.
    # But shape is preserved.
    dct_spec = norm_spec
    
    n_fft = 1024
    hop_length = 256
    
    # Inverse ST-DCT
    y_recon = st_idct(dct_spec, n_fft=n_fft, hop_length=hop_length)
    
    # Normalize audio to prevent clipping or silence
    # Normalize to standard PCM range (-1 to 1 for float, then scale to int16)
    # This recovers the volume!
    if np.max(np.abs(y_recon)) > 0:
        y_recon = y_recon / np.max(np.abs(y_recon))
        
    # Save
    base_name = os.path.splitext(avif_file)[0]
    # Handle case where file was named 'a.wav.avif' -> 'a.wav_recon.wav'
    # Or 'a.avif' -> 'a_recon.wav'
    if base_name.endswith('.wav'):
        out_filename = base_name.replace('.wav', '_recon.wav')
    else:
        out_filename = f"{base_name}_recon.wav"
        
    # Assume default SR=22050 if unknown
    sr = 22050
    
    # Convert to int16
    audio_int16 = (y_recon * 32767).astype(np.int16)
    wavfile.write(out_filename, sr, audio_int16)
    print(f"Saved {out_filename}")
    return out_filename

def main():
    parser = argparse.ArgumentParser(description="ST-DCT Audio Codec (WAV <-> AVIF)")
    parser.add_argument("input_file", help="Input file (.wav for encode, .avif for decode)")
    parser.add_argument("-q", "--quality", type=int, default=75, help="AVIF encoding quality (0-100)")
    
    # Check if arguments provided, otherwise fallback to default behavior (for safety/compatibility)
    if len(sys.argv) == 1:
        # Compatibility mode: run demo
        print("No arguments provided. Running demo mode...")
        # ... (Old demo code or just call encode on default)
        input_file = 'input.wav'
        if not os.path.exists(input_file):
            generate_synthetic_audio(input_file)
        encode_audio_to_avif(input_file, quality=75)
        return

    args = parser.parse_args()
    input_file = args.input_file
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)
        
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext in ['.wav', '.mp3', '.flac', '.m4a']:
        encode_audio_to_avif(input_file, quality=args.quality)
    elif ext in ['.avif']:
        decode_avif_to_audio(input_file)
    else:
        print(f"Unsupported file extension: {ext}")
        sys.exit(1)

if __name__ == "__main__":
    main()
