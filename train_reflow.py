# train_reflow.py (V4.2: Heatmap Vis, No F1)
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

warnings.filterwarnings("ignore")

from dataset import OsuManiaDataset, mania_collate_fn
from models.VAE import VAE
from models.ReflowDiT import ReflowDiT 
from models.Discriminator import ChartDiscriminator
from rectified_flow import RectifiedFlow

@torch.no_grad()
def run_inference_batch(reflow_model, vae_model, batch, steps=20):
    device = next(reflow_model.parameters()).device
    B = batch['encodec_features'].shape[0]
    n_keys = batch['n_keys'].squeeze(1)
    difficulty = batch['difficulty']
    
    x_t = torch.randn(B, vae_model.time_dim, vae_model.latent_dim, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t_val = i * dt
        t = torch.tensor([t_val] * B, device=device)
        v_pred = reflow_model(
            x_t, t, 
            batch['encodec_features'], 
            batch['extra_features_latent'],
            batch['extra_features_audio'],
            difficulty, n_keys
        )
        x_t = x_t + v_pred * dt
    
    # VAE Decode -> Logits -> Sigmoid -> Heatmap [B, T, K, C]
    final_probs = vae_model.decode_from_latent(x_t, n_keys)
    return final_probs

def visualize_heatmaps(reflow_model, vae_model, batch, epoch, output_dir, k=4):
    print(f"\n[Visualizing] Generating Heatmap Preview for Ep {epoch}...")
    reflow_model.eval(); vae_model.eval()
    vis_dir = os.path.join(output_dir, "reflow_heatmap_vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    k = min(k, batch['encodec_features'].shape[0])
    batch_k = {key: val[:k] for key, val in batch.items()}
    ho_true = batch_k['hit_objects']
    
    with torch.no_grad():
        ho_pred = run_inference_batch(reflow_model, vae_model, batch_k, steps=20)
        
        # Plot: Rows=Sample, Cols=[Gen Tap, GT Tap, Gen Hold, GT Hold]
        fig, axes = plt.subplots(k, 4, figsize=(16, 3 * k), dpi=100)
        cols = ["Gen Tap", "GT Tap", "Gen Hold", "GT Hold"]
        for ax, col in zip(axes[0], cols): ax.set_title(col)

        for i in range(k):
            # [T, K, 2]
            gen = ho_pred[i].cpu().numpy()
            gt = ho_true[i].cpu().numpy()
            
            # Tap (Ch0) - Greenish
            axes[i, 0].imshow(gen[..., 0].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='viridis')
            axes[i, 1].imshow(gt[..., 0].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='viridis')
            
            # Hold (Ch1) - Reddish
            axes[i, 2].imshow(gen[..., 1].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='magma')
            axes[i, 3].imshow(gt[..., 1].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='magma')
            
            for ax in axes[i]: ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"heatmap_epoch_{epoch:03d}.png"))
    plt.close()

def save_checkpoint(reflow_model, discriminator, optimizer_G, optimizer_D, scaler, scheduler, epoch, loss, checkpoint_dir, filename):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    state_dict = {
        'epoch': epoch,
        'reflow_model_state_dict': reflow_model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    if scheduler: state_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state_dict, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def calculate_adaptive_weight(loss_content, loss_adv, last_layer_weights, threshold=0.0):
    """
    计算自适应对抗权重 lambda = ||grad_content|| / ||grad_adv||
    """
    # 获取 content (MSE) loss 对最后一层的梯度
    grad_content = torch.autograd.grad(loss_content, last_layer_weights, retain_graph=True)[0]
    
    # 获取 adv (GAN) loss 对最后一层的梯度
    grad_adv = torch.autograd.grad(loss_adv, last_layer_weights, retain_graph=True)[0]
    
    # 计算梯度模长
    norm_content = torch.norm(grad_content)
    norm_adv = torch.norm(grad_adv)
    
    # 计算平衡系数
    # 加 1e-4 防止除以 0
    adaptive_weight = norm_content / (norm_adv + 1e-4)
    
    # 截断防止梯度爆炸 (通常限制在 0.0 ~ 1e4 之间)
    adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()
    
    return adaptive_weight

def train_one_epoch(reflow_model, discriminator, ae_model, dataloader, 
                    optimizer_G, optimizer_D, rf_instance, device, scaler, use_bf16, epoch, lambda_adv):
    reflow_model.train()
    discriminator.train()
    
    total_loss_g = 0.0
    total_loss_d = 0.0
    
    # 记录动态权重的平均值，用于监控
    total_adaptive_weight = 0.0
    
    pbar = tqdm(dataloader, desc=f"Train Ep {epoch}")
    
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        n_keys = batch['n_keys'].squeeze(1)
        difficulty = batch['difficulty']
        
        with torch.no_grad():
            x1 = ae_model.encode_to_latent(batch['hit_objects'], n_keys)
        x0 = torch.randn_like(x1)
        t_flow = torch.rand(x1.size(0), device=device)
        x_t, _ = rf_instance.create_flow(x1, t_flow, x0=x0)

        # ---------------------------
        # 1. Discriminator Step
        # ---------------------------
        optimizer_D.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16, enabled=use_bf16):
            real_latent = x1.permute(0, 2, 1)
            with torch.no_grad():
                v_pred_detached = reflow_model(
                    x_t, t_flow, batch['encodec_features'], 
                    batch['extra_features_latent'], batch['extra_features_audio'], 
                    difficulty, n_keys
                )
            t_batch = t_flow.view(-1, 1, 1)
            fake_latent = (x_t + (1 - t_batch) * v_pred_detached).permute(0, 2, 1)
            
            d_real = discriminator(real_latent, batch['encodec_features'], difficulty)
            d_fake = discriminator(fake_latent, batch['encodec_features'], difficulty)
            
            loss_d = torch.mean(torch.relu(1.0 - d_real)) + torch.mean(torch.relu(1.0 + d_fake))
        
        scaler.scale(loss_d).backward()
        scaler.step(optimizer_D)
        
        # ---------------------------
        # 2. Generator Step
        # ---------------------------
        optimizer_G.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16, enabled=use_bf16):
            v_pred = reflow_model(
                x_t, t_flow, batch['encodec_features'], 
                batch['extra_features_latent'], batch['extra_features_audio'], 
                difficulty, n_keys
            )
            
            # A. Weighted MSE Loss (Content Loss)
            # 计算 Note 能量权重: 1.0 + 5.0 * Energy (你之前说的5倍)
            target_energy = torch.norm(x1, dim=-1, keepdim=True)
            mse_weights = 1.0 + 5.0 * target_energy
            loss_mse = rf_instance.mse_loss(v_pred, x1, x0, weights=mse_weights)
            
            # B. Adversarial Loss (Raw)
            fake_latent_grad = (x_t + (1 - t_batch) * v_pred).permute(0, 2, 1)
            d_fake_g = discriminator(fake_latent_grad, batch['encodec_features'], difficulty)
            loss_adv = -torch.mean(d_fake_g)
            
            # C. Calculate Adaptive Weight
            # 我们希望 loss_adv 的梯度模长 == loss_mse 的梯度模长
            # 最后一层权重: reflow_model.output_proj.weight
            adaptive_w = calculate_adaptive_weight(
                loss_mse, loss_adv, reflow_model.output_proj.weight, threshold=0.0
            )
            
            # 最终 Loss = MSE + (Adaptive * Lambda) * Adv
            # lambda_adv 变成了一个"调节系数"，通常设为 1.0 或 0.5 即可，不用设太小
            loss_g = loss_mse + adaptive_w * lambda_adv * loss_adv

        scaler.scale(loss_g).backward()
        scaler.step(optimizer_G)
        scaler.update()

        total_loss_g += loss_g.item()
        total_loss_d += loss_d.item()
        total_adaptive_weight += adaptive_w.item()
        
        pbar.set_postfix({
            'G': f"{loss_g.item():.2f}", 
            'D': f"{loss_d.item():.2f}",
            'W': f"{adaptive_w.item():.2f}"
        })
        
    return total_loss_g / len(dataloader), total_loss_d / len(dataloader)

@torch.no_grad()
def validate_one_epoch(reflow_model, discriminator, ae_model, dataloader, rf_instance, device, use_bf16, epoch, output_dir, visualize_flag):
    reflow_model.eval(); discriminator.eval()
    total_loss, total_fool, total_score = 0.0, 0.0, 0.0
    first_batch = None
    
    pbar = tqdm(dataloader, desc=f"Valid Ep {epoch}")
    for i, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        if i == 0 and visualize_flag: first_batch = batch
        n_keys = batch['n_keys'].squeeze(1)
        difficulty = batch['difficulty']
        
        x1 = ae_model.encode_to_latent(batch['hit_objects'], n_keys)
        x0 = torch.randn_like(x1)
        t_flow = torch.rand(x1.size(0), device=device)
        x_t, _ = rf_instance.create_flow(x1, t_flow, x0=x0)
        
        with autocast(dtype=torch.bfloat16, enabled=use_bf16):
            v_pred = reflow_model(
                x_t, t_flow, batch['encodec_features'], 
                batch['extra_features_latent'], batch['extra_features_audio'], 
                difficulty, n_keys
            )
            loss = rf_instance.mse_loss(v_pred, x1, x0)
            
            fake_latent = (x_t + (1 - t_flow.view(-1,1,1)) * v_pred).permute(0, 2, 1)
            d_logits = discriminator(fake_latent, batch['encodec_features'], difficulty)
            
            total_fool += (d_logits > 0).float().mean().item()
            total_score += d_logits.mean().item()
        
        total_loss += loss.item()
        pbar.set_postfix({'MSE': f"{loss.item():.3f}", 'Fool': f"{total_fool/(i+1):.2f}"})

    avg_loss = total_loss / len(dataloader)
    fool_rate = total_fool / len(dataloader)
    d_score = total_score / len(dataloader)
    
    if visualize_flag and first_batch:
        visualize_heatmaps(reflow_model, ae_model, first_batch, epoch, output_dir)
        
    return avg_loss, fool_rate, d_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_reflow_gan")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--checkpoint_freq", type=int, default=5)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--ae_stats_file", type=str, default=None)
    parser.add_argument("--lambda_adv", type=float, default=0.5)
    
    args = parser.parse_args()
    device = torch.device(args.device)
    use_bf16 = args.use_bf16 and torch.cuda.is_bf16_supported()

    print(f"Loading VAE from {args.vae_checkpoint_path}...")
    vae_model = VAE().to(device)
    vae_ckpt = torch.load(args.vae_checkpoint_path, map_location=device)
    if 'model_state_dict' in vae_ckpt: vae_ckpt = vae_ckpt['model_state_dict']
    vae_model.load_state_dict(vae_ckpt, strict=False)
    vae_model.eval()
    for p in vae_model.parameters(): p.requires_grad = False
    print(f"VAE Scale Factor: {vae_model.scale_factor.item():.4f}")

    print("Initializing Generator & Discriminator...")
    reflow_model = ReflowDiT().to(device)
    discriminator = ChartDiscriminator().to(device)
    
    print("Loading Data...")
    train_ds = OsuManiaDataset(args.train_data_dir, mode='reflow')
    val_ds = OsuManiaDataset(args.val_data_dir, mode='reflow')
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=mania_collate_fn, num_workers=4, pin_memory=True)
    val_bs = max(1, args.batch_size // 2)
    val_loader = DataLoader(val_ds, val_bs, shuffle=False, collate_fn=mania_collate_fn, num_workers=4)

    if args.use_8bit_adam and HAS_BNB:
        print("Using 8-bit AdamW")
        optimizer_G = bnb.optim.AdamW8bit(reflow_model.parameters(), lr=args.lr, weight_decay=0.01)
        optimizer_D = bnb.optim.AdamW8bit(discriminator.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        print("Using standard AdamW")
        optimizer_G = optim.AdamW(reflow_model.parameters(), lr=args.lr, weight_decay=0.01)
        optimizer_D = optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.num_epochs)
    scaler = GradScaler(enabled=use_bf16)
    rf_instance = RectifiedFlow()
    
    os.makedirs(args.output_dir, exist_ok=True)
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        reflow_model.load_state_dict(ckpt['reflow_model_state_dict'])
        # Handle new D
        if 'discriminator_state_dict' in ckpt: discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        else: print("Notice: New Discriminator initialized.")
        
        # Handle optims
        if 'optimizer_G_state_dict' in ckpt: optimizer_G.load_state_dict(ckpt['optimizer_G_state_dict'])
        elif 'optimizer_state_dict' in ckpt: optimizer_G.load_state_dict(ckpt['optimizer_state_dict'])
        if 'optimizer_D_state_dict' in ckpt: optimizer_D.load_state_dict(ckpt['optimizer_D_state_dict'])
            
        if 'scaler_state_dict' in ckpt: scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'scheduler_state_dict' in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'loss' in ckpt: best_val_loss = ckpt['loss']
        print(f"Resumed at Epoch {start_epoch}")

    print("Start GAN Training...")
    for epoch in range(start_epoch, args.num_epochs + 1):
        loss_g, loss_d = train_one_epoch(
            reflow_model, discriminator, vae_model, train_loader, 
            optimizer_G, optimizer_D, rf_instance, device, scaler, use_bf16, epoch, 
            args.lambda_adv
        )
        val_loss, fool_rate, d_score = validate_one_epoch(
            reflow_model, discriminator, vae_model, val_loader, rf_instance, 
            device, use_bf16, epoch, args.output_dir, args.visualize
        )
        print(f"Ep {epoch} | G: {loss_g:.4f} D: {loss_d:.4f} | Val MSE: {val_loss:.4f} | Fool: {fool_rate:.2f} (Sc: {d_score:.2f})")
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(reflow_model, discriminator, optimizer_G, optimizer_D, scaler, scheduler, epoch, val_loss, args.output_dir, "reflow_gan_best.pth")
        if epoch % args.checkpoint_freq == 0:
            save_checkpoint(reflow_model, discriminator, optimizer_G, optimizer_D, scaler, scheduler, epoch, val_loss, args.output_dir, f"reflow_gan_epoch_{epoch:03d}.pth")

if __name__ == '__main__':
    main()