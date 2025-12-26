# slice_dataset.py
import os
import torch
import torchaudio
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

def slice_single_map(map_dir, output_dir, target_sample_rate, audio_seq_length, frame_resolution_ms, num_segments_per_map, min_audio_length_seconds):
    """
    切片单个谱面文件夹。
    """
    map_name = Path(map_dir).name
    try:
        original_map_index = int(map_name)
    except ValueError:
        print(f"  跳过 {map_dir}，目录名 '{map_name}' 不是纯数字。")
        return

    audio_path = os.path.join(map_dir, "audio.wav")
    timing_states_path = os.path.join(map_dir, "timing_states.pt")
    hit_objects_path = os.path.join(map_dir, "hit_objects.pt")
    difficulty_path = os.path.join(map_dir, "difficulty.json")

    # 检查必要文件是否存在
    if not all(os.path.exists(p) for p in [audio_path, timing_states_path, hit_objects_path, difficulty_path]):
        print(f"  跳过 {map_dir}，缺少必要文件。")
        return

    try:
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform = waveform

        # 加载谱面数据
        timing_states_full = torch.load(timing_states_path) # [T_full, 7] (假设已归一化)
        hit_objects_full = torch.load(hit_objects_path)     # [T_full, N_KEYS, 2]

        # 检查音频长度是否足够
        audio_duration_seconds = waveform.shape[1] / target_sample_rate
        if audio_duration_seconds < min_audio_length_seconds:
            print(f"  跳过 {map_dir}，音频长度 ({audio_duration_seconds:.2f}s) 小于最小要求 ({min_audio_length_seconds}s)。")
            return

        # 计算总帧数和目标片段帧数
        total_frames = timing_states_full.shape[0]
        frames_per_segment = int((audio_seq_length / target_sample_rate) * 1000 / frame_resolution_ms) # 1000ms/s

        # 计算音频总采样点数
        total_samples = waveform.shape[1]

        # 计算可以切片的起始采样点的最大索引 (确保片段不会越界)
        max_start_sample_idx = max(total_samples - audio_seq_length, 0)

        # 计算可以切片的起始帧的最大索引 (确保片段不会越界)
        max_start_frame_idx = max(total_frames - frames_per_segment, 0)

        # 确保有足够的空间进行切片
        if max_start_sample_idx <= 0 or max_start_frame_idx <= 0:
            print(f"  跳过 {map_dir}，音频或谱面长度不足以切出一个完整的 {audio_seq_length/target_sample_rate:.1f}s 片段。")
            return

        # 随机选择切片的起始采样点索引
        # np.random.randint 生成 [0, max_start_sample_idx) 范围内的随机整数
        start_sample_indices = np.random.randint(0, max_start_sample_idx + 1, size=num_segments_per_map)

        for i, start_sample_idx in enumerate(start_sample_indices):
            end_sample_idx = start_sample_idx + audio_seq_length
            # 提取音频片段
            segment_waveform = waveform[:, start_sample_idx:end_sample_idx]

            # 计算对应的起始和结束时间（秒）
            start_time_seconds = start_sample_idx / target_sample_rate
            end_time_seconds = end_sample_idx / target_sample_rate

            # 计算对应的起始和结束帧索引
            start_frame_idx = int(start_time_seconds * 1000 / frame_resolution_ms)
            end_frame_idx = int(end_time_seconds * 1000 / frame_resolution_ms)
            # 确保帧索引不越界
            start_frame_idx = min(start_frame_idx, max_start_frame_idx)
            end_frame_idx = min(end_frame_idx, total_frames)

            # 提取谱面片段
            segment_timing_states = timing_states_full[start_frame_idx:end_frame_idx, :] # [frames_per_segment, 7] (或更短)
            segment_hit_objects = hit_objects_full[start_frame_idx:end_frame_idx, :, :] # [frames_per_segment, N_KEYS, 2] (或更短)

            # 如果提取的帧数少于目标帧数，用零填充
            current_frames = segment_timing_states.shape[0]
            if current_frames < frames_per_segment:
                pad_frames = frames_per_segment - current_frames
                pad_shape_ts = [pad_frames, segment_timing_states.shape[1]]
                pad_shape_ho = [pad_frames, segment_hit_objects.shape[1], segment_hit_objects.shape[2]]
                padding_ts = torch.zeros(pad_shape_ts, dtype=segment_timing_states.dtype)
                padding_ho = torch.zeros(pad_shape_ho, dtype=segment_hit_objects.dtype)
                segment_timing_states = torch.cat([segment_timing_states, padding_ts], dim=0)
                segment_hit_objects = torch.cat([segment_hit_objects, padding_ho], dim=0)

            # --- 修改：生成新的文件夹名 ---
            new_subdir_index = original_map_index * 10 + i
            output_subdir = os.path.join(output_dir, str(new_subdir_index))
            os.makedirs(output_subdir, exist_ok=True)

            # 保存切片后的数据
            torchaudio.save(os.path.join(output_subdir, "audio.wav"), segment_waveform, target_sample_rate)
            torch.save(segment_timing_states, os.path.join(output_subdir, "timing_states.pt"))
            torch.save(segment_hit_objects, os.path.join(output_subdir, "hit_objects.pt"))
            # 复制 difficulty.json
            with open(difficulty_path, 'r', encoding='utf-8') as src_f, \
                 open(os.path.join(output_subdir, "difficulty.json"), 'w', encoding='utf-8') as dst_f:
                dst_f.write(src_f.read())

    except Exception as e:
        print(f"  处理 {map_dir} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="Slice osu!mania dataset into shorter segments.")
    parser.add_argument("input_dir", type=str, help="Input directory containing original map folders (e.g., data/train).")
    parser.add_argument("output_dir", type=str, help="Output directory to store sliced segments (e.g., data/train_sp).")
    parser.add_argument("--target_sample_rate", type=int, default=24000, help="Target sample rate for audio (default: 24000).")
    parser.add_argument("--audio_seq_length", type=int, default=240000, help="Length of each audio segment in samples (default: 240000 for 10s @ 24kHz).")
    parser.add_argument("--frame_resolution_ms", type=int, default=10, help="Frame resolution for timing_states and hit_objects (default: 10ms).")
    parser.add_argument("--num_segments_per_map", type=int, default=10, help="Number of random segments to extract from each original map (default: 10).")
    parser.add_argument("--min_audio_length_seconds", type=float, default=15.0, help="Minimum audio length in seconds for a map to be processed (default: 15.0).")

    args = parser.parse_args()

    print(f"参数: {args}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 获取所有原始谱面文件夹
    # 现在需要过滤出纯数字命名的文件夹
    all_subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    map_dirs = []
    for d in all_subdirs:
        try:
            int(d.name) # 尝试转换为整数
            map_dirs.append(d)
        except ValueError:
            continue # 如果转换失败，则忽略

    print(f"找到 {len(map_dirs)} 个原始谱面文件夹 (纯数字命名)。")

    # 设置随机种子以确保可复现性 (可选)
    # np.random.seed(42)

    # 遍历并切片
    for map_dir in tqdm(map_dirs, desc="切片进度"):
        slice_single_map(
            map_dir=map_dir,
            output_dir=output_dir,
            target_sample_rate=args.target_sample_rate,
            audio_seq_length=args.audio_seq_length,
            frame_resolution_ms=args.frame_resolution_ms,
            num_segments_per_map=args.num_segments_per_map,
            min_audio_length_seconds=args.min_audio_length_seconds
        )

    print(f"数据切片完成。结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
