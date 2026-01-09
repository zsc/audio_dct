# 基于短时离散余弦变换 (ST-DCT) 的音频编解码器

本项目实现了一个实验性的音频编解码器，利用短时离散余弦变换 (ST-DCT) 将音频编码为图像（AVIF 或 PNG），并能从图像解码回音频。通过巧妙利用图像通道，该编解码器在保持高压缩率的同时实现了良好的重建音质。

## 功能特性
- **音频转图像 (编码)**: 将 `.wav` 等音频文件转换为 `.avif` (有损) 或 `.png` (无损/高保真) 图像。
- **符号位映射**: 巧妙地将 DCT 系数的正部和负部映射到图像的不同颜色通道（红色和蓝色），有效保留了相位信息。
- **Mel 谱支持**: 支持 Mel 刻度频率压缩，在保持主观听感的同时显著降低图像高度。
- **Mu-law 压扩**: 使用 mu-law 算法对幅值进行非线性映射，增强小信号的保真度，提升听感。
- **元数据持久化**: 在图像 EXIF (Tag 270) 中自动保存 `max_val`、`n_dct`、压扩开关及原始音频长度等关键参数。

## 依赖要求
- Python 3.x
- `numpy`, `scipy`, `librosa`, `matplotlib`, `pillow`, `pillow-avif-plugin`

安装示例：
```bash
pip install numpy scipy librosa matplotlib pillow pillow-avif-plugin
```

## 使用方法

### 1. 音频编码为图像
```bash
python st_dct.py input.wav -q 90 -H 192
```
- `-q`, `--quality`: AVIF 压缩质量 (0-100)，默认 90。
- `-H`, `--height`: 图像高度（Mel 频段数或 DCT 频率轴大小），默认 192。
- `--png`: 使用 16 位 PNG 格式存储（更高精度，但文件较大）。
- `--no-mel`: 使用线性频率刻度而非 Mel 刻度。
- `--no-mulaw`: 禁用 mu-law 压扩。

### 2. 图像解码为音频
```bash
python st_dct.py input.avif
```
- 支持输入 `.avif` 或 `.png`。
- 脚本将自动从 EXIF 元数据中读取参数并重建音频。
- 输出文件默认为 `*_recon.wav`。

### 3. 质量评估 (PSNR)
使用自带脚本对比原始音频与重建音频的信噪比：
```bash
python calculate_psnr.py original.wav reconstructed.wav
```

## 可视化与对比
- `compare.html`: 一个简单的 Web 界面，用于在浏览器中快速对比多段音频的音质。
- `index.html`: 频谱可视化演示页面。

## 工作原理
1. **编码阶段**:
   - 对音频信号执行分帧和短时离散余弦变换 (ST-DCT-II)。
   - **分路处理**: 将 DCT 谱拆分为正部 (`max(0, x)`) 和负部 (`max(0, -x)`)。
   - **特征映射**: 对幅值应用 mu-law 压扩及可选的 Mel 滤波器组降维。
   - **图像生成**: 将正部存入 **红色通道**，负部存入 **蓝色通道**，绿色通道保持为 0 (YUV444 采样)。
   - **元数据嵌入**: 将缩放因子等元数据写入 EXIF。
2. **解码阶段**:
   - 从图像的红蓝通道中提取幅值，并合并回带符号的 DCT 谱。
   - 逆转 Mel 变换（如适用）及 mu-law 映射。
   - 执行逆 ST-DCT 并通过重叠相加法 (Overlap-Add) 还原波形。

## 文件说明
- `st_dct.py`: 核心编解码逻辑。
- `calculate_psnr.py`: PSNR 评估工具。
- `compare.html` & `index.html`: 可视化与展示工具。
