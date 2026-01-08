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

def generate_synthetic_audio(filename, duration=3.0, sr=16000):
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

def encode_audio_to_avif(audio_file, height=80, quality=75):
    """Encode audio to AVIF image."""
    print(f"Encoding {audio_file} to AVIF (Height={height}, Q={quality})...")
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    
    n_fft = 1024
    hop_length = n_fft // 4
    
    dct_spec = st_dct(y, n_fft=n_fft, hop_length=hop_length)
    
    # Normalization to 0-255
    max_val = np.max(np.abs(dct_spec))
    if max_val == 0:
        max_val = 1.0
        
    norm_spec = (dct_spec / max_val + 1) / 2
    img_data = (norm_spec * 255).astype(np.uint8)
    
    img = Image.fromarray(img_data, mode='L')
    
    # Embed max_val in EXIF (ImageDescription - Tag 270)
    exif = img.getexif()
    exif[270] = str(max_val)
    
    base_name = os.path.splitext(audio_file)[0]
    avif_filename = f"{base_name}.avif"
    
    img.save(avif_filename, format='AVIF', quality=quality, exif=exif)
    print(f"Saved {avif_filename} (Shape: {img_data.shape[1]}x{img_data.shape[0]}, MaxVal={max_val:.4f})")
    return avif_filename

def decode_avif_to_audio(avif_file):
    """Decode AVIF image to audio."""
    print(f"Decoding {avif_file} to WAV...")
    img = Image.open(avif_file)
    if img.mode != 'L':
        img = img.convert('L')
        
    img_data = np.array(img).astype(np.float32)
    
    # Detect n_fft from image height
    n_fft = img_data.shape[0]
    hop_length = n_fft // 4
    
    print(f"Detected n_fft={n_fft} from image height.")
    
    # Try to read max_val from EXIF
    exif = img.getexif()
    max_val = 1.0
    if exif and 270 in exif:
        try:
            max_val = float(exif[270])
            print(f"Restored max_val from EXIF: {max_val:.4f}")
        except ValueError:
            print("Failed to parse max_val from EXIF, using default 1.0")
    else:
        print("No max_val found in EXIF, using default 1.0")

    # Map [0, 255] back to [-1, 1]
    norm_spec = (img_data / 255.0) * 2 - 1
    
    # Restore amplitude
    dct_spec = norm_spec * max_val
    
    # Inverse ST-DCT
    y_recon = st_idct(dct_spec, n_fft=n_fft, hop_length=hop_length)
    
    # No auto-normalization here if we trust max_val, 
    # but we might still clip if artifacts introduced peaks.
    # Let's clip to safe range instead of re-normalizing.
    # Or just let wavfile.write handle clipping (it wraps or clips depending on impl).
    # Safest is to clip.
    # y_recon = np.clip(y_recon, -1.0, 1.0) # Actually, raw float output might be > 1 if original was.
    # But usually audio is -1 to 1.
    
    base_name = os.path.splitext(avif_file)[0]
    if base_name.endswith('.wav'):
        out_filename = base_name.replace('.wav', '_recon.wav')
    else:
        out_filename = f"{base_name}_recon.wav"
        
    sr = 16000
    # Convert to int16
    audio_int16 = (np.clip(y_recon, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(out_filename, sr, audio_int16)
    print(f"Saved {out_filename}")
    return out_filename

def main():
    parser = argparse.ArgumentParser(description="ST-DCT Audio Codec (WAV <-> AVIF)")
    parser.add_argument("input_file", help="Input file (.wav for encode, .avif for decode)")
    parser.add_argument("-q", "--quality", type=int, default=75, help="AVIF encoding quality (0-100)")
    parser.add_argument("-H", "--height", type=int, default=80, help="Image height (number of frequency bins, default 1024)")
    
    if len(sys.argv) == 1:
        print("No arguments provided. Running demo mode...")
        input_file = 'input.wav'
        if not os.path.exists(input_file):
            generate_synthetic_audio(input_file)
        encode_audio_to_avif(input_file, height=80, quality=75)
        return

    args = parser.parse_args()
    input_file = args.input_file
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)
        
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext in ['.wav', '.mp3', '.flac', '.m4a']:
        encode_audio_to_avif(input_file, height=args.height, quality=args.quality)
    elif ext in ['.avif']:
        decode_avif_to_audio(input_file)
    else:
        print(f"Unsupported file extension: {ext}")
        sys.exit(1)

if __name__ == "__main__":
    main()
