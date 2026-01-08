# Short-Time DCT-II Audio Codec

This project implements an experimental audio codec that uses Short-Time Discrete Cosine Transform (ST-DCT) to encode audio into AVIF images and decode them back to audio.

## Features
- **Audio to Image (Encode)**: Converts `.wav` audio into `.avif` images using ST-DCT spectrograms.
- **Image to Audio (Decode)**: Reconstructs `.wav` audio from `.avif` spectrograms using Inverse ST-DCT and Overlap-Add.
- **Compression Control**: Use the `-q` flag to control AVIF compression quality (0-100).
- **Visualization**: (Legacy) The script can also generate spectrogram visualizations if modified or in previous versions.

## Requirements
- Python 3.x
- `numpy`, `scipy`, `librosa`, `matplotlib`
- `pillow`, `pillow-avif-plugin`

## Usage

### Encode Audio to AVIF
```bash
python st_dct.py input.wav -q 80
```
- Input: `input.wav`
- Output: `input.avif`
- `-q`: Quality (default 75). Lower values = smaller file size, more artifacts.

### Decode AVIF to Audio
```bash
python st_dct.py input.avif
```
- Input: `input.avif`
- Output: `input_recon.wav`

## How it Works
1. **Encoding**:
   - Computes Short-Time DCT-II.
   - Normalizes coefficients to 0-255 range.
   - Saves as an 8-bit Grayscale AVIF image.
2. **Decoding**:
   - Loads the AVIF image.
   - Maps pixel values back to signed float coefficients.
   - Performs Inverse ST-DCT with Overlap-Add to reconstruct the waveform.
   - Normalizes amplitude (volume information is relative).

## Files
- `st_dct.py`: Main codec script.
- `compare.html`: Helper to compare original and reconstructed audio.
