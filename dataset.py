# dataset.py
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
import warnings

warnings.filterwarnings("ignore")

def mania_collate_fn(batch):
    if not batch: return {}
    keys = batch[0].keys()
    collated_batch = {key: [] for key in keys}
    K_MAX = 9 
    for sample in batch:
        for key, value in sample.items():
            if key == 'hit_objects':
                hit_obj = value
                pad_size = K_MAX - hit_obj.shape[1]
                if pad_size > 0:
                    padding = torch.zeros((hit_obj.shape[0], pad_size, hit_obj.shape[2]), dtype=hit_obj.dtype)
                    hit_obj = torch.cat([hit_obj, padding], dim=1)
                collated_batch[key].append(hit_obj)
            else:
                collated_batch[key].append(value)
    for key, values in collated_batch.items():
        collated_batch[key] = torch.stack(values)
    return collated_batch

def _normalize_difficulty(diff_data, exclude_cs=False):
    PARAM_RANGES = { 'HPDrainRate': (0.0, 10.0), 'CircleSize': (1.0, 9.0), 'OverallDifficulty': (0.0, 10.0), 'ApproachRate': (0.0, 10.0), 'SliderMultiplier': (0.4, 3.6), 'SliderTickRate': (0.5, 8.0) }
    if exclude_cs: ORDERED_PARAMS = ['HPDrainRate', 'OverallDifficulty', 'ApproachRate', 'SliderMultiplier', 'SliderTickRate']
    else: ORDERED_PARAMS = ['HPDrainRate', 'CircleSize', 'OverallDifficulty', 'ApproachRate', 'SliderMultiplier', 'SliderTickRate']
    normalized = []
    for param in ORDERED_PARAMS:
        val = diff_data.get(param, (PARAM_RANGES[param][0] + PARAM_RANGES[param][1]) / 2.0)
        p_min, p_max = PARAM_RANGES[param]
        norm_val = (val - p_min) / (p_max - p_min) if (p_max - p_min) != 0 else 0.5
        normalized.append(np.clip(norm_val, 0.0, 1.0))
    return torch.tensor(normalized, dtype=torch.float32)

class OsuManiaDataset(Dataset):
    def __init__(self, root_dir, mode='reflow'):
        self.root_dir = root_dir
        self.mode = mode
        self.map_seq_length = 1000 
        
        self.sample_dirs = [
            os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else 999999) 
            if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()
        ]
        print(f"Dataset mode: '{self.mode}', Samples: {len(self.sample_dirs)}")

    def __len__(self): return len(self.sample_dirs)

    def compute_dense_features(self, timing_states, seq_len):
        bpm, sv, kiai = timing_states[:, 0], timing_states[:, 1], timing_states[:, 6]
        dt = 10.0 
        phases, cur = [], 0.0
        for t in range(seq_len):
            idx = t if t < len(bpm) else -1
            beat_len = bpm[idx].item()
            if beat_len <= 1.0: beat_len = 500.0 
            phases.append(cur)
            cur += dt / beat_len
            if cur >= 1.0: cur -= 1.0
        phases = torch.tensor(phases, dtype=torch.float32)
        beat_sin = torch.sin(2 * torch.pi * phases)
        beat_cos = torch.cos(2 * torch.pi * phases)
        if len(kiai) < seq_len:
            pad = seq_len - len(kiai)
            kiai = torch.nn.functional.pad(kiai, (0, pad))
            sv = torch.nn.functional.pad(sv, (0, pad))
        else:
            kiai, sv = kiai[:seq_len], sv[:seq_len]
        sv_norm = torch.clamp(sv / 4.0, -1.0, 1.0)
        return torch.stack([beat_sin, beat_cos, kiai, sv_norm], dim=0)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        # 1. Hit Objects -> Heatmap Conversion
        hit_obj_path = os.path.join(sample_dir, "hit_objects.pt")
        try:
            hit_objects_raw = torch.load(hit_obj_path)
        except:
            hit_objects_raw = torch.zeros(self.map_seq_length, 4, 2)
        
        if hit_objects_raw.shape[0] < self.map_seq_length:
            hit_objects_raw = torch.nn.functional.pad(hit_objects_raw, (0, 0, 0, 0, 0, self.map_seq_length - hit_objects_raw.shape[0]))
        elif hit_objects_raw.shape[0] > self.map_seq_length: 
            hit_objects_raw = hit_objects_raw[:self.map_seq_length]

        # === Gaussian Heatmap Generation (Smoother Version) ===
        ho_np = hit_objects_raw.numpy()
        heatmap = np.zeros_like(ho_np)
        
        # [修改] 使用较大的 Sigma (50ms)，保证波形饱满且梯度平滑
        # 且保留了唯一的最大值 (在 d=0 处)
        sigma = 5.0 

        K = ho_np.shape[1]
        
        for k in range(K):
            # Channel 0: Tap
            tap_track = ho_np[:, k, 0]
            if np.sum(tap_track) > 0:
                inv_mask = (tap_track < 0.5).astype(float)
                # 计算到最近 Note 的距离
                dist = distance_transform_edt(inv_mask)
                # 标准高斯，不加增益，保证峰值严格为 1.0
                heatmap[:, k, 0] = np.exp(-0.5 * (dist / sigma) ** 2)
            
            # Channel 1: Hold
            hold_track = ho_np[:, k, 1]
            if np.sum(hold_track) > 0:
                # Hold 不需要距离变换，直接模糊边缘即可
                blurred = gaussian_filter1d(hold_track, sigma=sigma, mode='constant')
                # 只有这里需要 Clip，因为卷积会叠加
                heatmap[:, k, 1] = np.clip(blurred, 0.0, 1.0)
        
        hit_objects = torch.from_numpy(heatmap).float()
        # ======================================================

        # 2. Difficulty
        diff_path = os.path.join(sample_dir, "difficulty.json")
        try:
            with open(diff_path, 'r', encoding='utf-8') as f:
                diff_data = json.load(f)
        except:
            diff_data = {}
            
        n_keys = torch.LongTensor([int(diff_data.get('CircleSize', 4))])

        if self.mode == 'ae':
            return { "hit_objects": hit_objects, "n_keys": n_keys }
        
        elif self.mode == 'reflow':
            # 3. EnCodec Features
            feat_path = os.path.join(sample_dir, "encodec_features.pt")
            if os.path.exists(feat_path):
                encodec_features = torch.load(feat_path)
            else:
                encodec_features = torch.zeros(128, 750)
            t_audio = encodec_features.shape[-1]
            
            # 4. Signal Features
            timing_path = os.path.join(sample_dir, "timing_states.pt")
            try:
                timing_states = torch.load(timing_path)
            except:
                timing_states = torch.zeros(self.map_seq_length, 7)
                
            if timing_states.dim() > 2: timing_states = timing_states.squeeze()
            
            extra_features_raw = self.compute_dense_features(timing_states, self.map_seq_length)
            
            # A. For Latent (Length 50)
            extra_features_latent = torch.nn.functional.interpolate(
                extra_features_raw.unsqueeze(0), size=50, mode='linear', align_corners=False
            ).squeeze(0)

            # B. For Audio (Length ~750)
            extra_features_audio = torch.nn.functional.interpolate(
                extra_features_raw.unsqueeze(0), size=t_audio, mode='linear', align_corners=False
            ).squeeze(0)

            difficulty = _normalize_difficulty(diff_data, exclude_cs=True)

            return {
                "encodec_features": encodec_features,
                "extra_features_latent": extra_features_latent,
                "extra_features_audio": extra_features_audio,
                "hit_objects": hit_objects,
                "difficulty": difficulty,
                "n_keys": n_keys,
            }