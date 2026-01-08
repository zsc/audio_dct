import numpy as np
import scipy.fft as fft
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import librosa
import os
import sys

def generate_synthetic_audio(filename, duration=3.0, sr=22050):
    """Generate a synthetic audio file with a sine sweep and some harmonics."""
    t = np.linspace(0, duration, int(sr * duration))
    # Sine sweep from 440Hz to 4400Hz
    sweep = np.sin(2 * np.pi * np.linspace(440, 4400, len(t)) * t)
    # Add some static tones
    tone1 = 0.5 * np.sin(2 * np.pi * 1000 * t)
    tone2 = 0.3 * np.sin(2 * np.pi * 2000 * t)
    
    audio = sweep + tone1 + tone2
    # Normalize to 16-bit PCM range
    audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
    wavfile.write(filename, sr, audio)
    print(f"Generated {filename}")

def st_dct(y, n_fft=2048, hop_length=512, window='hann'):
    """Compute Short-Time DCT-II."""
    # Frame the signal
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)
    
    # Apply window function
    win = librosa.filters.get_window(window, n_fft).reshape(-1, 1)
    frames = frames * win
    
    # Compute DCT-II for each frame
    # norm='ortho' makes it a truly orthogonal transform
    dct_out = fft.dct(frames, type=2, axis=0, norm='ortho')
    
    return dct_out

def main():
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = 'input.wav'
        if not os.path.exists(audio_file):
            generate_synthetic_audio(audio_file)
    
    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} not found.")
        sys.exit(1)
    
    # Load audio
    print(f"Loading {audio_file}...")
    y, sr = librosa.load(audio_file, sr=None)
    
    # Parameters
    n_fft = 1024
    hop_length = 256
    
    # Compute ST-DCT
    print("Computing ST-DCT...")
    dct_spec = st_dct(y, n_fft=n_fft, hop_length=hop_length)
    
    # Plotting
    print("Saving spectrogram...")
    plt.figure(figsize=(12, 6))
    
    # Create a custom colormap: Blue -> Black -> Red
    # 'cyan' for negative, 'black' for zero, 'magenta' or 'red' for positive can also be cool.
    # Let's stick to a classic Blue-Black-Red.
    cdict = {
        'red':   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
        'green': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    }
    # Or simply use LinearSegmentedColormap.from_list
    colors_list = [(0, 0, 1), (0, 0, 0), (1, 0, 0)] # Blue, Black, Red
    cmap_name = 'blue_black_red'
    cm = colors.LinearSegmentedColormap.from_list(cmap_name, colors_list, N=256)

    # Scaling limit for better contrast
    limit = np.max(np.abs(dct_spec)) * 0.1  
    
    img = librosa.display.specshow(
        dct_spec, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='log',
        cmap=cm,
        vmin=-limit,
        vmax=limit
    )
    
    # plt.colorbar(img, label='Amplitude') # Removed as per request
    plt.title(f'Short-Time DCT-II (Signed, Log-Scale): {os.path.basename(audio_file)}')
    plt.tight_layout()
    plt.savefig('spectrogram.png', facecolor='black') # Make border black too? Maybe not necessary, but looks cool.
    # Resetting figure facecolor to white just in case, or we can just stick to default for the outer part.
    # Actually let's keep the figure background white for readability of labels.
    plt.close()
    
    # Generate index.html
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ST-DCT Visualization</title>
    <style>
        body {{ font-family: sans-serif; display: flex; flex-direction: column; align-items: center; background: #f0f0f0; padding: 20px; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 100%; max-width: 1000px; }}
        img {{ width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }}
        audio {{ margin: 20px 0; width: 100%; }}
        code {{ background: #eee; padding: 2px 4px; border-radius: 4px; }}
        .legend {{ margin-top: 10px; font-size: 0.9em; color: #555; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Short-Time DCT-II Visualization</h1>
        <p>Processed File: <code>{os.path.abspath(audio_file)}</code></p>
        <audio controls src="{audio_file}"></audio>
        <div class="spectrogram">
            <h3>Spectrogram (DCT-II, Log-Frequency Scale)</h3>
            <img src="spectrogram.png" alt="ST-DCT Spectrogram">
            <p class="legend"><strong>Color Legend:</strong> <span style="color: blue;">Blue</span> = Negative, <strong>Black</strong> = Zero, <span style="color: red;">Red</span> = Positive. (Signed amplitude, not absolute/dB)</p>
        </div>
    </div>
</body>
</html>
"""
    with open('index.html', 'w') as f:
        f.write(html_content)
    print(f"Generated index.html and spectrogram.png for {audio_file}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
