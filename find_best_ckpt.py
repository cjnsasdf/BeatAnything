import os
import torch
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.signal import find_peaks

# 项目引用
from models.VAE import VAE
from models.ReflowDiT import ReflowDiT
from dataset import OsuManiaDataset, mania_collate_fn

# ==============================================================================
# 🎛️ 后处理超参数
# ==============================================================================
HYPER_PARAMETERS = {
    "snap_config": [
        (1,  4.0, 3.0),
        (2,  3.0, 2.5),
        (3,  3.0, 2.5),
        (4,  2.0, 2.0),
        (6,  1.5, 1.5),
        (8,  1.2, 1.5),
    ],
    # 只取能量最强的 Top 5% 作为 Note，压制生成密度
    "density_percentile": 85, 
    "min_threshold": 0.4,
    "min_dist": 2
}

# ==============================================================================

def load_timing_points(sample_dir):
    try:
        json_files = [f for f in os.listdir(sample_dir) if f.endswith('.json') and 'difficulty' not in f]
        if json_files:
            with open(os.path.join(sample_dir, json_files[0]), 'r', encoding='utf-8') as f:
                return json.load(f)['TimingPoints']
    except: pass
    return [{'time': 0, 'beatLength': 400.0, 'uninherited': 1}]

def generate_grid_weight(timing_points, total_frames, frame_ms=10.0):
    weight_map = np.ones(total_frames, dtype=np.float32)
    seq_len = total_frames
    red_lines = [tp for tp in timing_points if tp['uninherited']]
    if not red_lines: return weight_map
    sorted_tp = sorted(red_lines, key=lambda x: x['time'])
    total_ms = total_frames * frame_ms

    def add_gaussian_batch(centers, sigma, amp):
        centers = centers[(centers >= 0) & (centers < seq_len)]
        if len(centers) == 0: return
        radius = int(3 * sigma)
        x_local = np.arange(-radius, radius + 1)
        gaussian_kernel = np.exp(-0.5 * (x_local / sigma) ** 2) * amp
        for c in centers:
            start = int(c - radius); end = int(c + radius + 1)
            k_start, k_end = 0, len(gaussian_kernel)
            if start < 0: k_start = -start; start = 0
            if end > seq_len: k_end = len(gaussian_kernel) - (end - seq_len); end = seq_len
            if start < end: weight_map[start:end] += gaussian_kernel[k_start:k_end]

    for i in range(len(sorted_tp)):
        tp = sorted_tp[i]
        start_time = tp['time']
        beat_len = tp['beatLength']
        if i < len(sorted_tp) - 1: end_time = sorted_tp[i+1]['time']
        else: end_time = total_ms
        if beat_len <= 0: continue
        for divisor, amp, sigma in HYPER_PARAMETERS['snap_config']:
            interval_ms = beat_len / divisor
            if interval_ms < 30.0: continue
            num_beats = int((end_time - start_time) / interval_ms)
            if num_beats <= 0: continue
            beat_offsets = np.arange(num_beats + 1) * interval_ms
            beat_frames = (start_time + beat_offsets) / frame_ms
            add_gaussian_batch(beat_frames, sigma, amp)
    return weight_map

# [修复] 正确计算 GT 密度 (从 Heatmap 还原脉冲)
def calculate_gt_density(heatmap_np, window_size=100):
    # heatmap_np: [T, K, 2] (Gaussian Blurred)
    tap_sum = np.sum(heatmap_np[..., 0], axis=1) # [T]
    
    # 必须寻峰！不能直接 > 0.5，因为高斯波很宽
    # GT 的峰值是 1.0，所以 height=0.5 足够安全
    peaks, _ = find_peaks(tap_sum, height=0.5, distance=2)
    
    impulse = np.zeros_like(tap_sum)
    impulse[peaks] = 1.0 # 还原为单点脉冲
    
    # 卷积平滑 (NPS)
    kernel = np.ones(window_size)
    density = np.convolve(impulse, kernel, mode='same')
    return density

def calculate_nps_curve(heatmap_np, timing_points=None, window_size=100):
    """计算 Gen 密度"""
    T, K, C = heatmap_np.shape
    if timing_points:
        gw = generate_grid_weight(timing_points, T)
        heatmap_np = heatmap_np * gw[:, None, None]
    
    tap_sum = np.sum(heatmap_np[..., 0], axis=1)
    
    valid_vals = tap_sum[tap_sum > 0.01]
    if len(valid_vals) == 0: return np.zeros(T)
        
    perc = HYPER_PARAMETERS['density_percentile']
    adaptive_thresh = np.percentile(valid_vals, perc)
    final_thresh = max(HYPER_PARAMETERS['min_threshold'], adaptive_thresh)
    
    peaks, _ = find_peaks(tap_sum, height=final_thresh, distance=HYPER_PARAMETERS['min_dist'])
    
    impulse = np.zeros(T)
    impulse[peaks] = 1.0
    kernel = np.ones(window_size)
    density = np.convolve(impulse, kernel, mode='same')
    return density

def evaluate_batch(reflow_model, vae_model, batch, dataset, batch_indices, device, steps=10):
    B = batch['encodec_features'].shape[0]
    n_keys = batch['n_keys'].squeeze(1).to(device)
    difficulty = batch['difficulty'].to(device)
    
    x_t = torch.randn(B, vae_model.time_dim, vae_model.latent_dim, device=device)
    dt = 1.0 / steps
    
    with torch.no_grad():
        for i in range(steps):
            t_val = torch.tensor([i * dt] * B, device=device)
            v = reflow_model(
                x_t, t_val, 
                batch['encodec_features'].to(device),
                batch['extra_features_latent'].to(device),
                batch['extra_features_audio'].to(device),
                difficulty, n_keys
            )
            x_t = x_t + v * dt
        heatmap_batch = vae_model.decode_from_latent(x_t, n_keys).cpu().numpy()

    gen_densities = []
    for i in range(B):
        sample_idx = batch_indices[i]
        sample_dir = dataset.sample_dirs[sample_idx]
        tps = load_timing_points(sample_dir)
        d = calculate_nps_curve(heatmap_batch[i], timing_points=tps)
        gen_densities.append(d)
        
    return gen_densities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_optimized")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading VAE...")
    vae = VAE().to(device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    if isinstance(vae_ckpt, dict) and 'model_state_dict' in vae_ckpt: vae_ckpt = vae_ckpt['model_state_dict']
    vae.load_state_dict(vae_ckpt, strict=False)
    vae.eval()
    
    dataset = OsuManiaDataset(args.val_data_dir, mode='reflow')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mania_collate_fn)
    batch = next(iter(loader))
    batch_indices = list(range(args.batch_size))
    
    print("Calculating GT Density...")
    gt_densities = []
    gt_objects = batch['hit_objects'].numpy()
    for i in range(args.batch_size):
        # [Fix] 使用修复后的 GT 计算逻辑
        d = calculate_gt_density(gt_objects[i]) 
        gt_densities.append(d)
        
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "reflow_gan_epoch_*.pth")))
    try: ckpts.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    except: pass
    
    print(f"Found {len(ckpts)} checkpoints. Evaluating...")
    
    reflow = ReflowDiT().to(device)
    results = []
    
    for ckpt_path in tqdm(ckpts):
        ckpt_name = os.path.basename(ckpt_path)
        try:
            sd = torch.load(ckpt_path, map_location=device)
            if 'reflow_model_state_dict' in sd: reflow.load_state_dict(sd['reflow_model_state_dict'])
            else: continue
            reflow.eval()
            
            gen_densities = evaluate_batch(reflow, vae, batch, dataset, batch_indices, device)
            
            err = 0
            for i in range(args.batch_size):
                err += np.mean((gen_densities[i] - gt_densities[i]) ** 2)
            avg_err = err / args.batch_size
            results.append((ckpt_name, avg_err, gen_densities))
            
        except Exception as e:
            print(f"Error evaluating {ckpt_name}: {e}")
            
    results.sort(key=lambda x: x[1])
    top_results = results[:args.top_n]
    
    print("\nTop Checkpoints (Optimized):")
    for name, score, _ in top_results:
        print(f"  {name}: {score:.4f}")
        
    print("Plotting...")
    fig, axes = plt.subplots(args.batch_size, args.top_n, figsize=(5 * args.top_n, 2 * args.batch_size), sharex=True, sharey=True)
    if args.top_n == 1: axes = axes[:, None]
    
    for col, (name, score, gen_data) in enumerate(top_results):
        axes[0, col].set_title(f"{name}\nErr: {score:.3f}", fontsize=10, fontweight='bold')
        for row in range(args.batch_size):
            ax = axes[row, col]
            # GT (Black Area)
            ax.fill_between(range(1000), 0, gt_densities[row], color='black', alpha=0.3, label='GT')
            # Gen (Red Line)
            ax.plot(gen_data[row], color='red', linewidth=1.5, label='Gen')
            if col == 0: ax.set_ylabel(f"S{row}", fontsize=8)
            ax.grid(True, alpha=0.2)
            # 统一 Y 轴，NPS 一般不超过 15
            ax.set_ylim(0, 15) 
            
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "optimized_comparison_v2.png")
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()