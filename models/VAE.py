# models/VAE.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.norm2(self.conv2(self.act(self.norm1(self.conv1(x))))))

class VAE(nn.Module):
    def __init__(self, T_input=1000, K_max=9, C=2, latent_dim=32):
        super().__init__()
        self.T_input, self.K_max, self.C = T_input, K_max, C
        self.latent_dim = latent_dim
        self.time_dim = T_input // 20 
        self.register_buffer('scale_factor', torch.tensor(1.0))

        self.encoder = nn.Sequential(
            nn.Conv1d(K_max*C, 64, 3, padding=1),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.SiLU(), ResBlock(128),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.BatchNorm1d(256), nn.SiLU(), ResBlock(256),
            nn.Conv1d(256, 512, 5, stride=5, padding=0), nn.BatchNorm1d(512), nn.SiLU(), ResBlock(512),
        )
        self.to_moments = nn.Conv1d(512, latent_dim * 2, 1)
        self.n_keys_embed = nn.Embedding(K_max + 1, latent_dim)
        
        self.decoder_init = nn.Conv1d(latent_dim, 512, 1)
        self.decoder_blocks = nn.Sequential(
            ResBlock(512), nn.ConvTranspose1d(512, 256, 5, stride=5, padding=0),
            nn.BatchNorm1d(256), nn.SiLU(), ResBlock(256),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.SiLU(), ResBlock(128),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.SiLU(), ResBlock(64),
        )
        self.decoder_final = nn.Conv1d(64, K_max * C, 3, padding=1)

    def encode(self, x):
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        h = self.encoder(x)
        moments = self.to_moments(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def decode(self, z, n_keys):
        keys_embed = self.n_keys_embed(n_keys).unsqueeze(-1)
        z = z - keys_embed
        x = self.decoder_init(z)
        x = self.decoder_blocks(x)
        x = self.decoder_final(x)
        B = x.shape[0]
        x = x.view(B, self.K_max, self.C, self.T_input).permute(0, 3, 1, 2)
        
        # [核心修复] 使用 masked_fill 将无效区域设为极小值 (-1e4)
        # 这样 Sigmoid(-1e4) 就会接近 0.0，而不是 0.5
        mask = torch.arange(self.K_max, device=x.device)[None, None, :, None] < n_keys[:, None, None, None]
        x = x.masked_fill(~mask, -1e4)
        
        return x

    def forward(self, x, n_keys):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_logits = self.decode(z, n_keys)
        return recon_logits, mean, logvar

    def encode_to_latent(self, x, n_keys):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar) 
        z = z * self.scale_factor
        z = z + self.n_keys_embed(n_keys).unsqueeze(-1)
        return z.permute(0, 2, 1) 

    def decode_from_latent(self, z, n_keys):
        z = z.permute(0, 2, 1) 
        z = z / self.scale_factor
        logits = self.decode(z, n_keys)
        return torch.sigmoid(logits)