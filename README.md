# 基于短时离散余弦变换 (ST-DCT) 的音频编解码器

本项目实现了一个实验性的音频编解码器，利用短时离散余弦变换 (ST-DCT) 将音频编码为 AVIF 图像，并能从图像解码回音频。

## 功能特性
- **音频转图像 (编码)**: 将 `.wav` 等音频文件转换为 `.avif` 图像。
- **图像转音频 (解码)**: 通过逆 ST-DCT 和重叠相加法 (Overlap-Add) 从 `.avif` 图像重建音频。
- **可变图像高度**: 编码时可指定图像高度（对应频段数，类似 Mel 频段），默认为 80。
- **压缩控制**: 使用 `-q` 参数控制 AVIF 压缩质量。
- **自动检测**: 解码时自动从图像高度识别 DCT 窗口大小。

## 依赖要求
- Python 3.x
- `numpy`, `scipy`, `librosa`, `matplotlib`
- `pillow`, `pillow-avif-plugin`

## 使用方法

### 1. 音频编码为 AVIF
```bash
python st_dct.py input.wav -q 80 -H 128
```
- 输入: `input.wav`
- 输出: `input.avif`
- `-q`: 压缩质量 (0-100)，默认 75。
- `-H`: 图像高度 (频段数)，默认 80。

### 2. AVIF 解码为音频
```bash
python st_dct.py input.avif
```
- 输入: `input.avif`
- 输出: `input_recon.wav`

## 工作原理
1. **编码阶段**:
   - 计算短时离散余弦变换 (ST-DCT-II)。
   - 将 DCT 系数归一化并映射到 0-255 的 8 位灰度范围。
   - 使用 AVIF 格式进行高效的有损压缩存储。
2. **解码阶段**:
   - 读取 AVIF 图像像素。
   - 将 0-255 映射回有符号的浮点数系数。
   - 执行逆 ST-DCT 并通过重叠相加法还原波形。
   - 自动进行振幅归一化以保证音量。

## 文件说明
- `st_dct.py`: 核心编解码脚本。
- `compare.html`: 用于在浏览器中对比原始音频与重建音频的工具。
- `README.md`: 项目说明文档。