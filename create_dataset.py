# create_dataset.py

import os
import zipfile
import tempfile
import shutil
import argparse
import json
from glob import glob
from tqdm import tqdm
import threading
import concurrent.futures

# 依赖项检查
try:
    import torch
    from pydub import AudioSegment
    import numpy as np
    from PIL import Image
    import imageio.v2 as imageio
    import osu2json
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖: pip install torch numpy Pillow imageio pydub")
    exit(1)

# FFmpeg 路径配置
FFMPEG_BIN_PATH = "./ffmpeg/bin" 
if not os.path.isdir(FFMPEG_BIN_PATH) or not os.path.exists(os.path.join(FFMPEG_BIN_PATH, "ffmpeg.exe")):
    # 尝试检查系统路径，如果不强制依赖本地 ffmpeg 文件夹
    import shutil as shell_shutil
    if shell_shutil.which("ffmpeg") is None:
        print("="*50); print("错误：FFmpeg 未找到！"); print(f"请确保 FFMPEG_BIN_PATH ('{FFMPEG_BIN_PATH}') 是正确的,"); print("或者将 FFmpeg 的 bin 目录添加到系统环境变量 Path 中。"); print("="*50)
else:
    os.environ["PATH"] += os.pathsep + FFMPEG_BIN_PATH

def get_mania_keys(osu_file_path):
    """快速解析 .osu 文件以获取 CircleSize (Key 数)。"""
    try:
        with open(osu_file_path, 'r', encoding='utf-8') as f:
            in_difficulty_section = False
            for line in f:
                if line.strip() == "[Difficulty]": in_difficulty_section = True; continue
                if in_difficulty_section:
                    if line.startswith("CircleSize:"): return int(float(line.strip().split(':')[1]))
                    if line.strip().startswith("["): break
    except Exception: return 99
    return 99

def find_audio_path(osu_file_path):
    with open(osu_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("AudioFilename:"): return line.strip().split(':', 1)[1].strip()
    return None

def process_osz_archive(osz_path, base_output_dir, lock, pair_idx_counter, target_sr=24000):
    created_pairs_in_osz = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(osz_path, 'r') as zip_ref: zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile: return 0, f"Skipped (bad zip file)"
        
        osu_files = glob(os.path.join(temp_dir, '**', '*.osu'), recursive=True)
        if not osu_files: return 0, "Skipped (no .osu files)"
        
        for osu_file in osu_files:
            # 过滤非 4K-9K 的谱面
            if get_mania_keys(osu_file) > 9: continue
            
            with lock:
                current_idx = pair_idx_counter[0]
                pair_idx_counter[0] += 1
            
            pair_dir = os.path.join(base_output_dir, str(current_idx))
            
            try:
                # 1. 检查模式
                with open(osu_file, 'r', encoding='utf-8') as f:
                    if "Mode: 3" not in f.read(): continue
                
                # 2. 查找并处理音频
                audio_filename = find_audio_path(osu_file)
                if not audio_filename: continue
                
                src_audio_path = os.path.join(os.path.dirname(osu_file), audio_filename)
                if not os.path.exists(src_audio_path): continue
                
                os.makedirs(pair_dir, exist_ok=True)
                
                # --- [优化] 音频处理逻辑 ---
                # Pydub 会自动处理 mp3, ogg, wav 等各种输入格式
                sound = AudioSegment.from_file(src_audio_path)
                
                # 强制转换为单声道 (set_channels(1)) 和目标采样率 (set_frame_rate)
                # 直接导出为 WAV，避免中间 MP3 压缩损失
                sound = sound.set_channels(1).set_frame_rate(target_sr)
                sound.export(os.path.join(pair_dir, "audio.wav"), format="wav")
                # -------------------------

                # 3. 处理谱面文件
                dest_osu_path = os.path.join(pair_dir, "beatmap.osu")
                shutil.copy(osu_file, dest_osu_path)
                
                # osu -> json
                json_path = osu2json.osu_to_json(dest_osu_path)
                with open(json_path, 'r', encoding='utf-8') as f: beatmap_data = json.load(f)
                
                # 提取难度
                difficulty_data = beatmap_data.get("Difficulty", {})
                with open(os.path.join(pair_dir, "difficulty.json"), 'w', encoding='utf-8') as f: 
                    json.dump(difficulty_data, f, indent=4)
                
                # json -> tensor (Image)
                tensor_temp_dir = os.path.join(pair_dir, 'tensors')
                osu2json.json_to_tensors(json_path, tensor_temp_dir)
                
                # 处理 Tensor (Load Images -> Numpy -> Torch)
                tensor_a_np = imageio.imread(os.path.join(tensor_temp_dir, "a_timing_states.tif"))
                n_keys = int(difficulty_data.get('CircleSize', 4))
                n_frames = tensor_a_np.shape[0]
                
                tensor_b_np = np.zeros((n_keys, n_frames, 2), dtype=np.uint8)
                for k in range(n_keys):
                    for c in range(2):
                        # 查找对应的图片文件
                        fname = next(f for f in os.listdir(tensor_temp_dir) if f.startswith(f"b_key{k}_channel{c}"))
                        tensor_b_np[k, :, c] = np.array(Image.open(os.path.join(tensor_temp_dir, fname))).squeeze()
                
                # 保存最终 Tensor
                # 注意：这里保存的是原始 Timing States 数值，这对后续的 Signal-based Dataset 至关重要
                tensor_a_pt = torch.from_numpy(tensor_a_np).float().unsqueeze(-1)
                tensor_b_pt = torch.from_numpy(tensor_b_np).float() / 255.0
                tensor_b_pt = tensor_b_pt.permute(1, 0, 2)
                
                torch.save(tensor_a_pt, os.path.join(pair_dir, "timing_states.pt"))
                torch.save(tensor_b_pt, os.path.join(pair_dir, "hit_objects.pt"))
                
                # 清理临时文件
                os.remove(json_path)
                os.remove(dest_osu_path)
                shutil.rmtree(tensor_temp_dir)
                
                created_pairs_in_osz += 1
                
            except Exception as e:
                # 如果出错，清理该样本文件夹
                if os.path.exists(pair_dir): shutil.rmtree(pair_dir)
                continue
                
    return created_pairs_in_osz, f"Processed, found {created_pairs_in_osz} pairs."

def main():
    parser = argparse.ArgumentParser(description="osu!mania .osz dataset creation script.")
    parser.add_argument("input_dir", type=str, help="Folder containing .osz files.")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to store the processed dataset.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads (default: number of CPU cores).")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target audio sample rate (default: 24000).")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'"); return
        
    os.makedirs(args.output_dir, exist_ok=True)
    osz_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".osz")]
    
    if not osz_files:
        print(f"No .osz files found in '{args.input_dir}'."); return
        
    pair_idx_counter = [0]
    lock = threading.Lock()
    total_pairs_created = 0

    print(f"Starting dataset creation. Target Sample Rate: {args.sample_rate}Hz")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_osz_archive, osz_path, args.output_dir, lock, pair_idx_counter, args.sample_rate): osz_path for osz_path in osz_files}
        with tqdm(total=len(osz_files), desc="Overall Progress") as pbar:
            for future in concurrent.futures.as_completed(futures):
                osz_path = futures[future]
                try:
                    created_count, message = future.result()
                    total_pairs_created += created_count
                    pbar.set_postfix_str(f"'{os.path.basename(osz_path)[:20]}...': {message}")
                except Exception as e:
                    pbar.set_postfix_str(f"'{os.path.basename(osz_path)[:20]}...': Error ({e})")
                pbar.update(1)
    
    print(f"\nDataset creation complete. Total pairs created: {total_pairs_created}")
if __name__ == '__main__':
    main()