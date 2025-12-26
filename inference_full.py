# inference_full.py (Fixed Dimension Bug)
import os
import torch
import torchaudio
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import EncodecModel

# 导入模型定义
from models.VAE import VAE
from models.ReflowDiT import ReflowDiT
# 导入 Dataset 辅助函数
from dataset import _normalize_difficulty
# 导入后处理
from post_process import heatmap_to_hitobjects, export_osu

class AudioProcessor:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        
    def load(self, path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform

def compute_full_timing_signal(timing_points, total_ms, frame_ms=10.0):
    """
    计算全曲的 Timing Signal (BPM, Beat Sin/Cos, etc.)
    """
    total_frames = int(total_ms / frame_ms) + 100
    
    # [BPM, SV, Meter, Vol, Set, Idx, Kiai]
    timing_tensor = torch.zeros(total_frames, 7)
    timing_tensor[:, 0] = 500.0 # Default BPM
    timing_tensor[:, 1] = 1.0   # Default SV
    
    sorted_tp = sorted(timing_points, key=lambda x: x['time'])
    for i, tp in enumerate(sorted_tp):
        start_frame = max(0, int(tp['time'] / frame_ms))
        if i < len(sorted_tp) - 1:
            end_frame = int(sorted_tp[i+1]['time'] / frame_ms)
        else:
            end_frame = total_frames
        
        if start_frame >= total_frames: break
        
        if tp['uninherited']: # BPM
            timing_tensor[start_frame:, 0] = tp['beatLength']
        else: # SV
            sv = -100.0 / tp['beatLength'] if tp['beatLength'] < 0 else 1.0
            timing_tensor[start_frame:, 1] = sv
            
        kiai = 1.0 if (tp['effects'] & 1) else 0.0
        timing_tensor[start_frame:end_frame, 6] = kiai

    # Compute Signals
    bpm_vals = timing_tensor[:, 0]
    sv_vals = timing_tensor[:, 1]
    kiai_vals = timing_tensor[:, 6]
    
    dt = frame_ms
    phases = []
    current_phase = 0.0
    
    for t in range(total_frames):
        beat_len = bpm_vals[t].item()
        if beat_len <= 1.0: beat_len = 500.0
        phases.append(current_phase)
        current_phase += dt / beat_len
        if current_phase >= 1.0: current_phase -= 1.0
        
    phases = torch.tensor(phases)
    beat_sin = torch.sin(2 * torch.pi * phases)
    beat_cos = torch.cos(2 * torch.pi * phases)
    sv_norm = torch.clamp(sv_vals / 4.0, -1.0, 1.0)
    
    # [4, T]
    return torch.stack([beat_sin, beat_cos, kiai_vals, sv_norm], dim=0)

def load_encodec_model(device):
    model_id = "facebook/encodec_24khz"
    print(f"Loading EnCodec: {model_id}...")
    try:
        print(f"  -> HF failed. Trying ModelScope...")
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download(model_id)
            model = EncodecModel.from_pretrained(model_dir).to(device)
        except Exception as ms_e:
            raise ms_e
    except Exception as e:
        model = EncodecModel.from_pretrained(model_id).to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Full Song Inference")
    parser.add_argument("--json_path", type=str, required=True, help="Input .json")
    parser.add_argument("--audio_path", type=str, required=True, help="Input .wav/.mp3")
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--reflow_ckpt", type=str, required=True)
    parser.add_argument("--output_osu", type=str, default="output.osu")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--chunk_samples", type=int, default=240000) 
    
    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Load Resources
    print(f"Loading Metadata: {args.json_path}")
    with open(args.json_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    print(f"Loading Audio: {args.audio_path}")
    processor = AudioProcessor()
    full_waveform = processor.load(args.audio_path)
    total_samples = full_waveform.shape[1]
    
    # 2. Compute Global Timing
    total_ms = (total_samples / 24000) * 1000
    print("Computing global timing signals...")
    full_timing_signal = compute_full_timing_signal(meta_data['TimingPoints'], total_ms)

    # 3. Load Models
    print("Loading Models...")
    vae = VAE().to(device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    if isinstance(vae_ckpt, dict) and 'model_state_dict' in vae_ckpt: vae_ckpt = vae_ckpt['model_state_dict']
    vae.load_state_dict(vae_ckpt, strict=False)
    vae.eval()
    
    encodec = load_encodec_model(device)
    
    reflow = ReflowDiT().to(device)
    reflow_ckpt = torch.load(args.reflow_ckpt, map_location=device)
    if 'reflow_model_state_dict' in reflow_ckpt: reflow_ckpt = reflow_ckpt['reflow_model_state_dict']
    reflow.load_state_dict(reflow_ckpt)
    reflow.eval()

    # 4. Condition
    diff_data = meta_data['Difficulty']
    difficulty = _normalize_difficulty(diff_data, exclude_cs=True).unsqueeze(0).to(device)
    n_keys = torch.LongTensor([int(diff_data.get('CircleSize', 4))]).to(device)

    # 5. Extract Global EnCodec
    print("Extracting Global EnCodec Features...")
    with torch.no_grad():
        full_encodec_features = encodec.encoder(full_waveform.unsqueeze(0).to(device))
    print(f"Global Features Shape: {full_encodec_features.shape}")

    # 6. Chunk Inference
    full_heatmap = []
    
    CHUNK_DURATION_SEC = 10
    SAMPLES_PER_CHUNK = 24000 * CHUNK_DURATION_SEC
    FEAT_FRAMES_PER_CHUNK = SAMPLES_PER_CHUNK // 320 # 750
    TIMING_FRAMES_PER_CHUNK = CHUNK_DURATION_SEC * 100 # 1000
    
    total_chunks = int(np.ceil(full_encodec_features.shape[-1] / FEAT_FRAMES_PER_CHUNK))
    
    print(f"Starting Inference ({total_chunks} chunks)...")
    
    for i in tqdm(range(total_chunks)):
        # Slice EnCodec
        start_f = i * FEAT_FRAMES_PER_CHUNK
        end_f = min((i + 1) * FEAT_FRAMES_PER_CHUNK, full_encodec_features.shape[-1])
        chunk_encodec = full_encodec_features[:, :, start_f:end_f]
        
        current_feat_len = chunk_encodec.shape[-1]
        pad_len_feat = FEAT_FRAMES_PER_CHUNK - current_feat_len
        if pad_len_feat > 0:
            chunk_encodec = torch.nn.functional.pad(chunk_encodec, (0, pad_len_feat))
            
        # Slice Timing
        start_t = i * TIMING_FRAMES_PER_CHUNK
        end_t = start_t + TIMING_FRAMES_PER_CHUNK
        
        if start_t >= full_timing_signal.shape[1]:
            chunk_timing = torch.zeros(4, TIMING_FRAMES_PER_CHUNK)
        else:
            chunk_timing = full_timing_signal[:, start_t:end_t]
            if chunk_timing.shape[1] < TIMING_FRAMES_PER_CHUNK:
                pad_t = TIMING_FRAMES_PER_CHUNK - chunk_timing.shape[1]
                chunk_timing = torch.nn.functional.pad(chunk_timing, (0, pad_t))
        
        chunk_timing = chunk_timing.unsqueeze(0).to(device) # [1, 4, 1000]

        with torch.no_grad():
            # [FIX] 移除 .squeeze(0)，保持 [1, 4, T] 3D 形状
            extra_lat = torch.nn.functional.interpolate(chunk_timing, size=50, mode='linear')
            extra_aud = torch.nn.functional.interpolate(chunk_timing, size=FEAT_FRAMES_PER_CHUNK, mode='linear')
            
            x_t = torch.randn(1, 50, 32, device=device)
            dt = 1.0 / args.steps
            
            for s in range(args.steps):
                t_val = torch.tensor([s * dt], device=device)
                # chunk_encodec [1, 128, 750]
                # extra_aud [1, 4, 750]
                # 现在维度匹配了，可以 cat
                v = reflow(x_t, t_val, chunk_encodec, extra_lat, extra_aud, difficulty, n_keys)
                x_t = x_t + v * dt
                
            heatmap_chunk = vae.decode_from_latent(x_t, n_keys)
            
            if pad_len_feat > 0:
                valid_frames = int(1000 * (current_feat_len / FEAT_FRAMES_PER_CHUNK))
                heatmap_chunk = heatmap_chunk[:, :valid_frames, :, :]
            
            full_heatmap.append(heatmap_chunk.cpu())
            
    # 7. Stitch & Export
    print("Stitching and Exporting...")
    full_heatmap = torch.cat(full_heatmap, dim=1).squeeze(0).numpy()
    
    hit_objects = heatmap_to_hitobjects(
        full_heatmap, 
        meta_data['TimingPoints'], 
        n_keys=int(n_keys.item())
    )
    
    export_osu(meta_data, hit_objects, args.output_osu)

if __name__ == "__main__":
    main()