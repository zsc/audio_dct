# 基于短时离散余弦变换 (ST-DCT) 的音频编解码器

本项目实现了一个实验性的音频编解码器，利用短时离散余弦变换 (ST-DCT) 将音频编码为 AVIF 图像，并能从图像解码回音频。

## 功能特性
- **音频转图像 (编码)**: 将 `.wav` 等音频文件转换为 `.avif` 图像。
- **符号位保留**: 提取 DCT 符号并单独存储在 EXIF 中，大幅提升重建质量。
- **非负 Mel 谱**: 图像像素代表 DCT 振幅的绝对值（可经 Mel 映射），视觉更直观。
- **图像转音频 (解码)**: 通过逆 ST-DCT 和重叠相加法 (Overlap-Add) 从 `.avif` 图像重建音频。
- **元数据持久化**: 自动在图像元数据中保存缩放因子、窗口大小及符号信息。

## 依赖要求
- Python 3.x
- `numpy`, `scipy`, `librosa`, `matplotlib`
- `pillow`, `pillow-avif-plugin`

## 使用方法

### 1. 音频编码为 AVIF
```bash
python st_dct.py input.wav -q 90 -H 128
```
- 输入: `input.wav`
- 输出: `input.avif`
- `-q`: 压缩质量 (0-100)，默认 90。
- `-H`: 图像高度 (Mel 频段数)，默认 128。

### 2. AVIF 解码为音频
```bash
python st_dct.py input.avif
```
- 输入: `input.avif`
- 输出: `input_recon.wav`

## 工作原理
1. **编码阶段**:
   - 计算短时离散余弦变换 (ST-DCT-II)。
   - **符号提取**: 记录系数的正负号，使用 `np.packbits` 打包后经 `zlib` 压缩和 `base64` 编码。
   - **幅值处理**: 取 DCT 绝对值，并根据需要应用 Mel 滤波器组进行降维。
   - **元数据存储**: 将 `max_val`、`n_dct` 以及**压缩后的符号字符串**存入 AVIF 的 **EXIF (Tag 270)**。
   - **图像化**: 将幅值归一化到 0-255 并保存为 AVIF。
2. **解码阶段**:
   - 读取 AVIF 像素并还原为幅值。
   - 从 EXIF 中提取并解压**符号位**。
   - 将符号重新应用到幅值，还原带符号的 DCT 谱。
   - 执行逆 ST-DCT 并通过重叠相加法还原波形。

## 文件说明
- `st_dct.py`: 核心编解码脚本。
- `compare.html`: 用于在浏览器中对比原始音频与重建音频的工具。
- `README.md`: 项目说明文档。
