# BeatAnythingV5 - AI Osu!mania Beatmap Generator

**BeatAnythingV5** 是一个基于深度学习的 osu!mania 谱面生成系统。它利用 **Variational Autoencoder (VAE)** 将谱面压缩到潜在空间，并使用结合了对抗生成网络 (GAN) 损失的 **Rectified Flow (DiT)** 模型，根据音频特征（EnCodec）和难度信息生成高质量的谱面。

## ⚙️ 关键技术细节

1.  **ReflowDiT 模型**:
    *   采用 **Rectified Flow** 匹配，相比传统 DDPM 收敛更快，生成质量更高。
    *   引入 **Audio ResNet Adapter**，将 EnCodec 的高频特征与谱面节奏特征对齐。
    *   结合 **Discriminator** (ChartDiscriminator)，在 Flow Matching Loss 基础上增加 GAN Loss，显著提升了生成谱面的节奏感和清晰度。

2.  **后处理 (Post-Process)**:
    *   **高斯热力图还原**: 使用自适应阈值从 VAE 输出的热力图中提取 Note。
    *   **强规则对齐**: 包含 `snap_time` 算法，强制将生成的 Note 对齐到 1/1, 1/2, 1/4, 1/8 等节拍线上。
    *   **冲突检测**: 自动修复重叠的 Note 和过短的 Hold。

3.  **特征工程**:
    *   **Timing Signal**: 显式编码 BPM、Sin/Cos 相位、Kiai 和 SV (Slider Velocity) 作为条件输入。
    *   **Difficulty**: 标准化处理 CircleSize, OD, HP 等难度参数。

## 📁 目录结构

```text
BeatAnythingV5/
│  create_dataset.py       # [数据预处理] 步骤1: 解压 .osz，音频转码，生成基础 Tensor
│  slice_dataset.py        # [数据预处理] 步骤2: 将长谱面切片为短片段 (10s) 用于训练
│  preprocess_encodec.py   # [数据预处理] 步骤3: 预提取 EnCodec 音频特征
│  dataset.py              # PyTorch Dataset 定义 (含高斯热力图生成)
│  
│  train_vae.py            # [训练] 训练 VAE 模型 (压缩/解压谱面)
│  train_reflow.py         # [训练] 训练 Rectified Flow DiT 模型 (生成模型)
│  find_best_ckpt.py       # [评估] 评估最佳模型权重 (基于密度一致性)
│  rectified_flow.py       # Rectified Flow 核心逻辑
│  
│  scrape_beatmaps.py      # [工具] osu! 官网爬虫 (基于 Selenium)
│  prepareInfer.py         # [推理] 推理前置准备 (音频/元数据处理)
│  inference_full.py       # [推理] 全曲推理脚本
│  post_process.py         # [后处理] 热力图转 HitObjects，吸附对齐算法
│  osu2json.py             # [工具] .osu 与 .json/.tensor 互转工具
│  
└─models/
        Discriminator.py   # 判别器 (用于 Reflow 的 GAN Loss)
        ReflowDiT.py       # 生成模型 (Diffusion Transformer + Audio Adapter)
        VAE.py             # 变分自编码器 (1D ResNet 架构)
```

## 🛠️ 环境依赖

请确保安装了 Python 3.8+ 和 FFmpeg。

1.  **安装 Python 库**:
    ```bash
    pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118  # 根据你的 CUDA 版本调整
    pip install numpy pandas matplotlib tqdm pydub imageio pillow scipy selenium transformers bitsandbytes
    # 如果 transformers 加载 EnCodec 失败，可能需要 modelscope:
    pip install modelscope
    ```

2.  **配置 FFmpeg**:
    *   项目代码默认会在 `./ffmpeg/bin` 寻找 `ffmpeg.exe`。
    *   或者，请确保 `ffmpeg` 已添加到系统的环境变量 PATH 中。

3.  **浏览器驱动 (仅爬虫需要)**:
    *   下载对应版本的 Microsoft Edge WebDriver，并放置在 `webdriver/msedgedriver.exe`。

---

## 🚀 数据集准备流程

### 1. 搜刮谱面 (可选)
使用 `scrape_beatmaps.py` 从 osu! 官网下载 `.osz` 文件。
*   **注意**: 需要提供 `osu_cookies.json` (使用浏览器插件 Cookie-Editor 导出)。
```bash
python scrape_beatmaps.py 100 --output_dir ./osz_files --links_file beatmap_links.txt
```

### 2. 创建基础数据集
解压 `.osz`，转换音频为 24kHz 单声道，将谱面转换为 Tensor。
```bash
python create_dataset.py ./osz_files --output_dir ./data/raw --sample_rate 24000
```

### 3. 数据切片
将长谱面切分为固定长度的片段（默认 10秒/240000 采样点），以便于训练。
```bash
python slice_dataset.py ./data/raw ./data/train_sliced --audio_seq_length 240000
```

### 4. 预提取 EnCodec 特征
为了加快训练速度，预先计算所有音频切片的 EnCodec 特征。
```bash
python preprocess_encodec.py ./data/train_sliced
```

---

## 🧠 模型训练

### 第一阶段: 训练 VAE
VAE 用于将稀疏的谱面网格（HitObjects）压缩为密集的潜在向量（Latent Code）。

```bash
python train_vae.py \
  --train_data_dir ./data/train_sliced \
  --val_data_dir ./data/train_sliced \
  --output_dir ./checkpoints_vae \
  --batch_size 32 \
  --num_epochs 50 \
  --visualize
```
*   **注意**: 训练结束后，脚本会自动计算 `scale_factor` 并保存到 `vae_best.pth` 中。

### 第二阶段: 训练 Reflow DiT
训练生成模型，学习从噪声和音频条件中恢复 VAE 的潜在向量。本项目使用了 **Rectified Flow** 结合 **Adversarial Loss (GAN)** 进行训练。

```bash
python train_reflow.py \
  --train_data_dir ./data/train_sliced \
  --val_data_dir ./data/train_sliced \
  --vae_checkpoint_path ./checkpoints_vae/vae_best.pth \
  --output_dir ./checkpoints_reflow \
  --batch_size 16 \
  --num_epochs 200 \
  --lambda_adv 0.5 \
  --use_bf16  # 推荐开启 BF16 加速
```

### (可选) 寻找最佳 Checkpoint
通过比较生成的 Note Density (NPS) 与 Ground Truth 的一致性来筛选最佳模型。
```bash
python find_best_ckpt.py \
  --ckpt_dir ./checkpoints_reflow \
  --vae_ckpt ./checkpoints_vae/vae_best.pth \
  --val_data_dir ./data/train_sliced \
  --top_n 5
```

---

## 🎵 推理 (生成谱面)

推理需要一个参考的 `.osu` 文件（用于提供 BPM、Offset 和难度设置）和一个目标音频文件。

### 1. 准备推理数据
该脚本会将音频转换为标准格式，并提取元数据。
```bash
python prepareInfer.py "ReferenceMap.osu" --audio_path "TargetSong.mp3" --output_dir ./infer_temp
```

### 2. 执行全曲生成
```bash
python inference_full.py \
  --json_path ./infer_temp/beatmap_meta.json \
  --audio_path ./infer_temp/audio.wav \
  --vae_ckpt ./checkpoints_vae/vae_best.pth \
  --reflow_ckpt ./checkpoints_reflow/reflow_gan_best.pth \
  --steps 20 \
  --output_osu "Generated_Beatmap.osu"
```

---

