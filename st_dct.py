import numpy as np
import scipy.fft as fft
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import librosa
import os
import sys
import argparse
import zlib
import base64
from PIL import Image
import pillow_avif  # Ensures AVIF support is registered

def get_mel_basis(sr, n_dct, n_mels):
    """Generate Mel filter bank for DCT coefficients."""
    # Librosa expects n_fft for STFT, where bins = n_fft // 2 + 1
    # We want to match n_dct bins.
    # Set n_fft_librosa = 2 * n_dct => n_dct + 1 bins.
    # Drop the last bin (Nyquist) to match n_dct bins.
    n_fft_librosa = 2 * n_dct
    m = librosa.filters.mel(sr=sr, n_fft=n_fft_librosa, n_mels=n_mels)
    return m[:, :-1]

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

def mu_law_encode(x, mu=255):
    """Apply mu-law companding to input x in [0, 1]."""
    return np.log(1 + mu * x) / np.log(1 + mu)

def mu_law_decode(y, mu=255):
    """Inverse mu-law companding for input y in [0, 1]."""
    return ((1 + mu) ** y - 1) / mu

def encode_audio(audio_file, height=128, quality=75, use_mel=True, use_png=False, use_mulaw=False):
    """Encode audio to image (AVIF or 16-bit PNG) with optional Mel-scale frequency compression."""
    mode_str = "Mel Bands" if use_mel else "Linear Bins"
    fmt = "PNG (16-bit)" if use_png else f"AVIF (Q={quality})"
    print(f"Encoding {audio_file} to {fmt} ({mode_str}={height})...")
    
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    
    # If not using Mel, height is the n_dct itself
    if use_mel:
        n_dct = 1024
        n_mels = height
    else:
        n_dct = height
        
    hop_length = n_dct // 4
    
    # Pad audio (Reflect padding, like STFT center=True)
    original_length = len(y)
    pad_length = n_dct // 2
    
    if len(y) < pad_length:
        y_padded = np.pad(y, pad_length, mode='edge')
    else:
        y_padded = np.pad(y, pad_length, mode='reflect')
    
    # 1. Compute ST-DCT (Linear Freq)
    dct_spec = st_dct(y_padded, n_fft=n_dct, hop_length=hop_length)
    
    # Split into positive and negative parts
    pos_spec = np.maximum(0, dct_spec)
    neg_spec = np.maximum(0, -dct_spec)
    
    if use_mel:
        # 2. Apply Mel Filter Bank (Linear Freq -> Mel Bands)
        mel_basis = get_mel_basis(sr, n_dct, n_mels)
        pos_final = np.dot(mel_basis, pos_spec)
        neg_final = np.dot(mel_basis, neg_spec)
    else:
        pos_final = pos_spec
        neg_final = neg_spec

    # Stack positive and negative parts vertically
    final_spec = np.vstack((pos_final, neg_final))
    print(f"Final spec shape (stacked): {final_spec.shape}")
    
    # Normalization
    max_val = np.max(final_spec)
    if max_val == 0:
        max_val = 1.0
        
    norm_spec = final_spec / max_val
    
    if use_mulaw:
        norm_spec = mu_law_encode(norm_spec)
    
    base_name = os.path.splitext(audio_file)[0]
    
    # Embed metadata in EXIF
    # Tag 270: "max_val,n_dct,use_mel,use_mulaw,original_length,pad_length"
    meta_str = f"{max_val},{n_dct},{1 if use_mel else 0},{1 if use_mulaw else 0},{original_length},{pad_length}"
    
    if use_png:
        # 16-bit PNG
        img_data = (norm_spec * 65535).astype(np.uint16)
        img = Image.fromarray(img_data, mode='I;16')
        out_filename = f"{base_name}.png"
        
        # PNGInfo for metadata in PNG chunks
        from PIL.PngImagePlugin import PngInfo
        metadata = PngInfo()
        metadata.add_text("Description", meta_str)
        # Also try to add EXIF for compatibility if reader looks there
        exif = img.getexif()
        exif[270] = meta_str
        
        img.save(out_filename, format='PNG', pnginfo=metadata, exif=exif)
        print(f"Saved {out_filename} (Shape: {img_data.shape[1]}x{img_data.shape[0]}, MaxVal={max_val:.4f})")
        
    else:
        # 8-bit AVIF
        img_data = (norm_spec * 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')
        out_filename = f"{base_name}.avif"
        
        exif = img.getexif()
        exif[270] = meta_str
        
        img.save(out_filename, format='AVIF', quality=quality, exif=exif)
        print(f"Saved {out_filename} (Shape: {img_data.shape[1]}x{img_data.shape[0]}, MaxVal={max_val:.4f})")

    return out_filename

def decode_audio(image_file):
    """Decode image (AVIF or PNG) back to audio, automatically detecting Mel compression."""
    print(f"Decoding {image_file} to WAV...")
    img = Image.open(image_file)
    
    # Detect bit depth and normalize
    # Expecting [0, 1] range
    if img.mode == 'I;16' or img.mode == 'I':
        print("Detected 16-bit image.")
        img_data = np.array(img).astype(np.float32)
        norm_spec = img_data / 65535.0
    else:
        print(f"Detected 8-bit image (Mode: {img.mode}).")
        if img.mode != 'L':
            img = img.convert('L')
        img_data = np.array(img).astype(np.float32)
        norm_spec = img_data / 255.0
        
    full_height, n_frames = img_data.shape
    
    # Parse Metadata
    # Try EXIF first (AVIF standard, and we added it to PNG too)
    exif = img.getexif()
    meta_str = None
    
    if exif and 270 in exif:
        meta_str = exif[270]
    else:
        # Try PNG info if EXIF failed (common for PNG)
        if 'Description' in img.info:
            meta_str = img.info['Description']
            
    max_val = 1.0
    n_dct = 1024
    use_mel = True
    use_mulaw = False
    original_length = 0
    pad_length = 0
    
    if meta_str:
        try:
            # Format: "max_val,n_dct,use_mel,use_mulaw,original_length,pad_length"
            if '|' in meta_str:
                # Legacy handling attempt or just ignore suffix
                meta_str = meta_str.split('|')[0]
                
            meta = meta_str.split(',')
            max_val = float(meta[0])
            if len(meta) > 1:
                n_dct = int(meta[1])
            if len(meta) > 2:
                use_mel = int(meta[2]) == 1
            if len(meta) > 3:
                use_mulaw = int(meta[3]) == 1
            if len(meta) > 5:
                original_length = int(meta[4])
                pad_length = int(meta[5])
            print(f"Restored metadata: max_val={max_val:.4f}, n_dct={n_dct}, use_mel={use_mel}, use_mulaw={use_mulaw}, orig_len={original_length}, pad={pad_length}")
        except ValueError:
            print("Failed to parse metadata, using defaults.")
    else:
        print("No metadata found, using defaults.")

    hop_length = n_dct // 4

    # Inverse Companding
    if use_mulaw:
        norm_spec = mu_law_decode(norm_spec)

    # Restore amplitude
    processed_spec = norm_spec * max_val
    
    # Split back into positive and negative parts
    half_height = full_height // 2
    pos_processed = processed_spec[:half_height, :]
    neg_processed = processed_spec[half_height:, :]
    
    if use_mel:
        # Inverse Mel Projection
        sr = 16000
        n_mels = half_height
        mel_basis = get_mel_basis(sr, n_dct, n_mels)
        mel_inv = np.linalg.pinv(mel_basis)
        
        pos_recon = np.dot(mel_inv, pos_processed)
        neg_recon = np.dot(mel_inv, neg_processed)
        
        # Combine: pos - neg
        dct_spec = pos_recon - neg_recon
    else:
        dct_spec = pos_processed - neg_processed
    
    # Inverse ST-DCT
    y_recon = st_idct(dct_spec, n_fft=n_dct, hop_length=hop_length)
    
    # Crop padding if info available
    if original_length > 0:
        start = pad_length
        end = start + original_length
        if end <= len(y_recon):
            y_recon = y_recon[start:end]
        else:
            y_recon = y_recon[start:]

    # Clip to safe range
    y_recon = np.clip(y_recon, -1.0, 1.0)
    
    base_name = os.path.splitext(image_file)[0]
    if base_name.endswith('.wav'):
        out_filename = base_name.replace('.wav', '_recon.wav')
    else:
        out_filename = f"{base_name}_recon.wav"
        
    sr = 16000
    audio_int16 = (y_recon * 32767).astype(np.int16)
    wavfile.write(out_filename, sr, audio_int16)
    print(f"Saved {out_filename}")
    return out_filename

def main():
    parser = argparse.ArgumentParser(description="ST-DCT Audio Codec (WAV <-> AVIF/PNG)")
    parser.add_argument("input_file", help="Input file (.wav for encode, .avif/.png for decode)")
    parser.add_argument("-q", "--quality", type=int, default=90, help="AVIF encoding quality (0-100)")
    parser.add_argument("-H", "--height", type=int, default=128, help="Image height (Mel bands or DCT bins)")
    parser.add_argument("--mel", action="store_true", default=True, help="Use Mel-scale frequency compression (default: True)")
    parser.add_argument("--no-mel", action="store_false", dest="mel", help="Use linear frequency scale")
    parser.add_argument("--png", action="store_true", help="Use 16-bit PNG format instead of AVIF")
    parser.add_argument("--mulaw", action="store_true", help="Use mu-law companding")
    
    if len(sys.argv) == 1:
        print("No arguments provided. Running demo mode (Mel enabled, AVIF)...")
        input_file = 'input.wav'
        if not os.path.exists(input_file):
            # Generate dummy if needed, but assuming user has it or we can't do much.
            print("input.wav not found. Please provide an input file.")
            sys.exit(1)
        encode_audio(input_file, height=128, quality=90, use_mel=True)
        return

    args = parser.parse_args()
    input_file = args.input_file
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)
        
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext in ['.wav', '.mp3', '.flac', '.m4a']:
        encode_audio(input_file, height=args.height, quality=args.quality, use_mel=args.mel, use_png=args.png, use_mulaw=args.mulaw)
    elif ext in ['.avif', '.png']:
        decode_audio(input_file)
    else:
        print(f"Unsupported file extension: {ext}")
        sys.exit(1)

if __name__ == "__main__":
    main()
