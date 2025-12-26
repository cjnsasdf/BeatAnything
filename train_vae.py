# train_vae.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import OsuManiaDataset, mania_collate_fn
from models.VAE import VAE

def visualize_vae_heatmap(model, dataset, epoch, output_dir, device, k=4):
    """
    可视化 VAE 重建的热力图对比
    """
    model.eval()
    vis_dir = os.path.join(output_dir, "vae_vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    indices = np.random.choice(len(dataset), min(k, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(k, 4, figsize=(16, 3 * k), dpi=100)
    cols = ["GT Tap", "Recon Tap", "GT Hold", "Recon Hold"]
    if k == 1:
        for ax, col in zip(axes, cols): ax.set_title(col)
    else:
        for ax, col in zip(axes[0], cols): ax.set_title(col)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            batch = mania_collate_fn([sample])
            hit_objects = batch['hit_objects'].to(device) # [1, T, K, 2]
            n_keys = batch['n_keys'].squeeze(-1).to(device)
            
            recon_logits, _, _ = model(hit_objects, n_keys)
            recon_probs = torch.sigmoid(recon_logits) 
            
            gt = hit_objects[0].cpu().numpy()
            recon = recon_probs[0].cpu().numpy()
            
            if k == 1: ax_row = axes
            else: ax_row = axes[i]
            
            # Tap (Ch 0)
            ax_row[0].imshow(gt[..., 0].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='viridis')
            ax_row[1].imshow(recon[..., 0].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='viridis')
            
            # Hold (Ch 1)
            ax_row[2].imshow(gt[..., 1].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='magma')
            ax_row[3].imshow(recon[..., 1].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='magma')
            
            for ax in ax_row: ax.axis('off')

    save_path = os.path.join(vis_dir, f"heatmap_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Vis] Saved heatmap to {save_path}")

def train_one_epoch(model, dataloader, optimizer, device, kl_weight):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss() 
    
    pbar = tqdm(dataloader, desc="Training VAE")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        recon_logits, mean, logvar = model(batch['hit_objects'], batch['n_keys'].squeeze(-1))
        
        recon_map = torch.sigmoid(recon_logits)
        # 放大 Loss，让数值更易读
        recon_loss = criterion(recon_map, batch['hit_objects']) * 100 
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch['hit_objects'].shape[0]
        loss = recon_loss + kl_weight * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Recon': f"{recon_loss.item():.4f}", 'KL': f"{kl_loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device, kl_weight):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating VAE"):
            batch = {k: v.to(device) for k, v in batch.items()}
            recon_logits, mean, logvar = model(batch['hit_objects'], batch['n_keys'].squeeze(-1))
            
            recon_map = torch.sigmoid(recon_logits)
            recon_loss = criterion(recon_map, batch['hit_objects']) * 100
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch['hit_objects'].shape[0]
            loss = recon_loss + kl_weight * kl_loss
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_vae_heatmap")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = VAE().to(device)
    
    train_loader = DataLoader(OsuManiaDataset(args.train_data_dir, mode='ae'), batch_size=args.batch_size, shuffle=True, collate_fn=mania_collate_fn, num_workers=4)
    val_loader = DataLoader(OsuManiaDataset(args.val_data_dir, mode='ae'), batch_size=args.batch_size, shuffle=False, collate_fn=mania_collate_fn, num_workers=4)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, 0.0001)
        val_loss = validate_one_epoch(model, val_loader, device, 0.0001)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "vae_best.pth"))
            
        # [修改] 始终在第 1 个 epoch 可视化，之后每 5 个 epoch 可视化
        if args.visualize and (epoch == 1 or epoch % 5 == 0):
            visualize_vae_heatmap(model, val_loader.dataset, epoch, args.output_dir, device)

    print("Calculating global scale factor...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "vae_best.pth")))
    model.eval()
    means = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Calculating Scale"):
            m, _ = model.encode(batch['hit_objects'].to(device))
            means.append(m.cpu())
    
    all_means = torch.cat(means, 0)
    std = all_means.std().item()
    scale = 1.0 / std if std > 0 else 1.0
    print(f"Global Latent STD: {std:.6f} | Scale Factor: {scale:.6f}")
    
    model.scale_factor.fill_(scale)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "vae_best.pth"))
    print("Done.")

if __name__ == '__main__':
    main()