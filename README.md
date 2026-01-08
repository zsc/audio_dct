# Short-Time DCT-II Visualization

This project provides a Python script to compute and visualize the Short-Time Discrete Cosine Transform (ST-DCT) of an audio signal.

## Features
- **Short-Time DCT-II**: Computes DCT-II on framed audio signals (similar to STFT).
- **Signed Visualization**: Unlike standard spectrograms that use absolute values (dB), this visualization preserves the sign of the coefficients.
- **Custom Colormap**: 
  - <span style="color: blue;">**Blue**</span>: Negative coefficients
  - **Black**: Zero
  - <span style="color: red;">**Red**</span>: Positive coefficients
- **Log-Frequency Scaling**: The y-axis uses a logarithmic scale to better represent audio frequency perception.
- **HTML Report**: Generates an `index.html` file for easy viewing of the audio and its spectrogram.

## Requirements
- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`
- `librosa`

## Usage
To process the default synthetic audio:
```bash
python st_dct.py
```

To process a specific audio file:
```bash
python st_dct.py path/to/your/audio.wav
```

## Output
- `input.wav`: Generated synthetic audio (if no input provided).
- `spectrogram.png`: The signed ST-DCT spectrogram.
- `index.html`: Web-based visualization.
