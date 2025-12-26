# models/Discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioFeatureDownsampler(nn.Module):
    """
    用于将高频音频特征 (EnCodec, ~75Hz) 下采样到 谱面特征 (~5Hz)
    目标: 将 [B, 128, 750] 压缩为 [B, 32, 50]
    """
    def __init__(self, in_channels=128, out_channels=32, target_len=50):
        super().__init__()
        self.target_len = target_len
        
        # 两步下采样: 750 -> 250 -> 50
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=3, padding=2), # /3
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=5, stride=5, padding=2), # /5
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(out_channels)
        )
        
        self.final_pool = nn.AdaptiveAvgPool1d(target_len)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_pool(x)
        return x

class ChartDiscriminator(nn.Module):
    def __init__(self, latent_dim=32, audio_dim=128, diff_dim=5, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        
        # 1. 音频编码器
        self.audio_encoder = AudioFeatureDownsampler(
            in_channels=audio_dim, 
            out_channels=32, 
            target_len=seq_len
        )
        
        # 2. [新增] 难度编码器
        # 将 5个 难度标量映射为向量
        self.diff_encoder = nn.Sequential(
            nn.Linear(diff_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 32) # 输出 32 通道
        )
        
        # 3. 判别器主干 (1D CNN)
        # Input: Latent(32) + Audio(32) + Diff(32) = 96
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv1d(32 + 32 + 32, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2 (Downsample)
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3 (Downsample)
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4 (Output Score)
            nn.Conv1d(512, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_latent, audio_feat, difficulty):
        """
        x_latent: [B, 32, 50]
        audio_feat: [B, 128, 750]
        difficulty: [B, 5]
        """
        # A. 对齐音频 -> [B, 32, 50]
        audio_cond = self.audio_encoder(audio_feat)
        
        # B. 处理难度 -> [B, 32, 50]
        diff_emb = self.diff_encoder(difficulty) # [B, 32]
        diff_cond = diff_emb.unsqueeze(-1).expand(-1, -1, self.seq_len) # [B, 32, 1] -> [B, 32, 50]
        
        # C. 拼接 (Latent + Audio + Difficulty)
        # 判别器现在会检查：谱面是否符合音频？谱面是否符合难度？
        x_in = torch.cat([x_latent, audio_cond, diff_cond], dim=1) # -> [B, 96, 50]
        
        # D. 判别
        logits = self.net(x_in) 
        
        return torch.mean(logits, dim=[1, 2])
