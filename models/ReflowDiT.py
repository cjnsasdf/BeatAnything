# models/ReflowDiT.py (V4.5: With Audio ResNet Adapter)
import torch
import torch.nn as nn
import math

# ==============================================================================
# Helper Modules
# ==============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, C]
        seq_len = x.size(1)
        if seq_len > self.max_len: return x
        return x + self.pe[:seq_len, :].unsqueeze(0)

class TimeEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dtype != torch.float32: t = t.float()
        t = t * 1000
        half_dim = self.dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / (half_dim - 1)))
        args = t.unsqueeze(1) * inv_freq.unsqueeze(0)
        embedding = torch.cat((args.sin(), args.cos()), dim=-1)
        if self.dim % 2 != 0: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

# [新增] Audio Adapter: 提取深层节奏特征，对齐声学空间与谱面空间
class AudioAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.net = nn.Sequential(
            # Layer 1: Feature Extraction
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.SiLU(),
            
            # Layer 2: Refinement
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.SiLU(),

            # Layer 2: Refinement
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.SiLU()
        )
        
        # Residual connection projection
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # x: [B, C, T]
        identity = self.proj(x)
        return self.net(x) + identity

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, context_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            kdim=context_dim, 
            vdim=context_dim, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        x_norm = self.norm(x)
        ctx_norm = self.context_norm(context)
        attn_out, _ = self.multihead_attn(query=x_norm, key=ctx_norm, value=ctx_norm)
        return x + self.dropout(attn_out)

class DiTBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.cross_attn = CrossAttentionLayer(dim, context_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), 
            nn.GELU(), 
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
        # AdaLN Zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, context, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        
        # 1. Self Attn
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + gate_msa * attn_out
        
        # 2. Cross Attn
        x = self.cross_attn(x, context)
        
        # 3. FFN
        x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)
        return x

# ==============================================================================
# Main Model: ReflowDiT
# ==============================================================================

class ReflowDiT(nn.Module):
    def __init__(self, latent_dim=32, time_dim=50, transformer_dim=512, nhead=8, num_layers=12):
        super().__init__()
        
        # Audio Context Dimension: EnCodec(128) + Timing(4) = 132
        audio_in_dim = 128 + 4 
        
        # --- 1. Latent Side Setup ---
        self.pos_emb = SinusoidalPositionalEmbedding(transformer_dim, max_len=time_dim)
        # Latent(32) + Timing(4) = 36
        self.input_proj = nn.Conv1d(latent_dim + 4, transformer_dim, kernel_size=3, padding=1)
        
        # --- 2. Audio Side Setup (V4.5: With Adapter) ---
        # 使用 Adapter 替代简单的 Conv1d
        self.audio_adapter = AudioAdapter(audio_in_dim, transformer_dim)
        self.audio_pos_emb = SinusoidalPositionalEmbedding(transformer_dim, max_len=2000)
        
        # --- 3. Global Conditions ---
        self.time_embedder = TimeEmbedder(128)
        self.n_keys_embed = nn.Embedding(10, 32)
        
        # MLP Difficulty Projection
        self.difficulty_proj = nn.Sequential(
            nn.Linear(5, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )
        
        self.global_mlp = nn.Sequential(
            nn.Linear(128 + 32 + 64, transformer_dim), 
            nn.SiLU(), 
            nn.Linear(transformer_dim, transformer_dim)
        )
        
        # --- 4. Transformer Backbone ---
        self.blocks = nn.ModuleList([
            DiTBlock(transformer_dim, transformer_dim, nhead) 
            for _ in range(num_layers)
        ])
        
        # --- 5. Output Head ---
        self.final_norm = nn.LayerNorm(transformer_dim)
        self.output_proj = nn.Conv1d(transformer_dim, latent_dim, kernel_size=3, padding=1)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-Init for AdaLN and Output
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, x_t, t, encodec_features, extra_features_latent, extra_features_audio, difficulty, n_keys):
        """
        x_t: [B, 50, 32] (Sequence First)
        encodec_features: [B, 128, T] (Channel First)
        extra_features_latent: [B, 4, 50] (Channel First)
        extra_features_audio: [B, 4, T] (Channel First)
        """
        
        # === 1. Audio Context Processing ===
        # Concat Audio + Timing -> [B, 132, T]
        audio_in = torch.cat([encodec_features, extra_features_audio], dim=1)
        
        # Apply ResNet Adapter -> [B, 512, T]
        audio_emb = self.audio_adapter(audio_in)
        
        # Permute to Sequence First [B, T, 512] & Add PosEmb
        audio_context = self.audio_pos_emb(audio_emb.permute(0, 2, 1))
        
        # === 2. Latent Input Processing ===
        # x_t [B, 50, 32] -> Permute [B, 32, 50]
        x_t_perm = x_t.permute(0, 2, 1) 
        
        # Concat Latent + Timing -> [B, 36, 50]
        x_in = torch.cat([x_t_perm, extra_features_latent], dim=1)
        
        # Conv Proj -> [B, 512, 50] -> Permute [B, 50, 512]
        x = self.input_proj(x_in).permute(0, 2, 1)
        x = self.pos_emb(x)
        
        # === 3. Global Conditions ===
        t_emb = self.time_embedder(t)
        k_emb = self.n_keys_embed(n_keys.view(-1))
        d_emb = self.difficulty_proj(difficulty)
        
        # [B, 1, 512]
        c = self.global_mlp(torch.cat([t_emb, k_emb, d_emb], dim=-1)).unsqueeze(1)
        
        # === 4. Transformer Loop ===
        for block in self.blocks:
            x = block(x, audio_context, c)
            
        # === 5. Output ===
        x = self.final_norm(x).permute(0, 2, 1) # -> [B, 512, 50] (Channel First)
        return self.output_proj(x).permute(0, 2, 1) # -> [B, 50, 32] (Sequence First)