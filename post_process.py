import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import label
import math

# ==============================================================================
# 🎛️ 后处理参数配置
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
    "density_percentile": 85, 
    "min_threshold": 0.4,
    "min_dist": 2,
    "collision_gap": 3
}

# ==============================================================================
# 🛠️ 强规则过滤组件 (New)
# ==============================================================================

class HitObject:
    def __init__(self, line, keys=4):
        self.line = line
        parts = line.split(',')
        self.x = int(parts[0])
        self.y = int(parts[1])
        self.time = int(parts[2])
        self.type = int(parts[3])
        self.hitSound = int(parts[4])
        self.extras = parts[5] # endTime:hitSample
        
        # 计算轨道 Index (0 ~ keys-1)
        # osu! algorithm: floor(x * keys / 512)
        self.column = int(self.x * keys / 512)
        self.keys = keys
        
        # 解析 EndTime
        if self.type & 128: # Hold
            self.end_time = int(self.extras.split(':')[0])
        else:
            self.end_time = self.time # Tap 的结束时间等于开始时间

    def is_hold(self):
        return (self.type & 128) > 0

    def set_time(self, new_start, new_end=None):
        self.time = int(new_start)
        if new_end is not None and self.is_hold():
            self.end_time = int(new_end)
            # Update extras string
            extra_parts = self.extras.split(':')
            extra_parts[0] = str(self.end_time)
            self.extras = ":".join(extra_parts)
    
    def to_string(self):
        return f"{self.x},{self.y},{self.time},{self.type},{self.hitSound},{self.extras}"

def snap_time(timestamp, timing_points, divisors=[1, 2, 3, 4, 6, 8, 12, 16]):
    """
    将时间戳吸附到最近的有效节拍线上
    """
    # 1. 找到当前的 Timing Point (Red Line)
    # Timing Points 应该已经排序
    ref_tp = timing_points[0]
    for tp in timing_points:
        if tp['time'] <= timestamp + 5: # +5ms 容错
            if tp['uninherited']: # 只看红线
                ref_tp = tp
        else:
            break
            
    bpm_len = ref_tp['beatLength']
    offset = ref_tp['time']
    
    if bpm_len <= 0: return timestamp # 异常保护
    
    # 2. 计算最近的吸附点
    # 目标：找到 t = offset + N * (beat_len / div) 最接近 timestamp
    best_snap = timestamp
    min_diff = float('inf')
    
    # 允许的最大吸附误差 (ms)
    # 如果离最近的线都超过这个值，说明可能是摇摆节奏或变速过渡，保持原样
    MAX_SNAP_ERROR = 10.0 
    
    for div in divisors:
        step = bpm_len / div
        # 当前时间相对于 Offset 是第几拍
        raw_beat_idx = (timestamp - offset) / step
        rounded_beat_idx = round(raw_beat_idx)
        
        snapped_time = offset + rounded_beat_idx * step
        diff = abs(snapped_time - timestamp)
        
        if diff < min_diff:
            min_diff = diff
            best_snap = snapped_time
            
    if min_diff <= MAX_SNAP_ERROR:
        return int(best_snap)
    else:
        return int(timestamp)

def finalize_beatmap(hit_objects_str_list, timing_points, n_keys=4):
    """
    最终过滤流程：
    1. 强制 Grid Snapping (对齐)
    2. 去除重叠 (Overlap Removal)
    3. 保证最小间隔 (Min Gap)
    """
    # 1. 解析对象
    objects = []
    for line in hit_objects_str_list:
        try:
            obj = HitObject(line, n_keys)
            objects.append(obj)
        except: continue
        
    # 按时间排序
    objects.sort(key=lambda x: x.time)
    
    # 2. 全局吸附 (Snapping)
    # 对所有 Start Time 和 End Time 进行吸附
    for obj in objects:
        new_start = snap_time(obj.time, timing_points)
        obj.set_time(new_start)
        
        if obj.is_hold():
            new_end = snap_time(obj.end_time, timing_points)
            # 保证长条至少有长度
            if new_end <= new_start:
                new_end = new_start + int(timing_points[0]['beatLength'] / 4) # 默认给个 1/4 拍
            obj.set_time(new_start, new_end)

    # 3. 逐轨道处理重叠 (Per-Column Processing)
    final_objects = []
    columns = [[] for _ in range(n_keys)]
    
    # 分轨
    for obj in objects:
        if 0 <= obj.column < n_keys:
            columns[obj.column].append(obj)
            
    # 最小间隔 (osu! 实际上允许 1ms，但为了 AI 生成的稳定性，建议 10ms 左右)
    MIN_GAP = 50
    
    for col_objs in columns:
        if not col_objs: continue
        
        # 按时间排序
        col_objs.sort(key=lambda x: x.time)
        
        clean_objs = []
        if len(col_objs) > 0:
            prev = col_objs[0]
            
            for i in range(1, len(col_objs)):
                curr = col_objs[i]
                
                # 检查冲突
                # prev_end 必须 < curr_start
                # 考虑到 Gap: prev_end + MIN_GAP <= curr_start
                
                if prev.end_time + MIN_GAP > curr.time:
                    # 发生重叠！
                    # 策略：优先保留后一个 Note (curr)，截断前一个 Note (prev)
                    # 因为后一个 Note 通常代表新的节奏点，节奏点比长条尾巴重要
                    
                    target_prev_end = curr.time - MIN_GAP
                    
                    # 如果截断后，前一个 Note 长度变成负数或极短
                    if target_prev_end <= prev.time + MIN_GAP:
                        # 极端情况：两个 Note 几乎贴在一起（如 100ms 和 105ms）
                        # 策略：删除前一个 Note，或者将其合并（AI 很难处理合并，删除较安全）
                        # 这里选择：如果 prev 是 Tap，删除 prev；如果 prev 是 Hold，尝试缩短
                        pass # 逻辑往下走
                    
                    if prev.is_hold():
                        # 尝试截断 Hold
                        new_duration = target_prev_end - prev.time
                        if new_duration >= 30: # 还有得救
                            prev.set_time(prev.time, target_prev_end)
                            clean_objs.append(prev)
                        else:
                            # 救不了，退化为 Tap
                            # 即使退化为 Tap，也可能和 curr 冲突
                            # 如果冲突，丢弃 prev
                            if prev.time + MIN_GAP <= curr.time:
                                # 变成 Tap
                                line_parts = prev.line.split(',')
                                # x,192,time,1,0,0:0:0:0:
                                new_line = f"{line_parts[0]},{line_parts[1]},{prev.time},1,{line_parts[4]},0:0:0:0:"
                                new_obj = HitObject(new_line, n_keys)
                                clean_objs.append(new_obj)
                    else:
                        # prev 是 Tap
                        if prev.time + MIN_GAP <= curr.time:
                            clean_objs.append(prev)
                        # else: 丢弃 prev (太近了)
                else:
                    # 没有冲突
                    clean_objs.append(prev)
                
                prev = curr
            
            # 添加最后一个
            clean_objs.append(prev)
            
        final_objects.extend(clean_objs)
        
    # 重新按时间排序
    final_objects.sort(key=lambda x: x.time)
    
    return [obj.to_string() for obj in final_objects]

# ==============================================================================
# 原有逻辑
# ==============================================================================

def generate_grid_weight(timing_points, total_frames, frame_ms=10.0):
    # ... (保持原样) ...
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
            start = int(np.floor(c - radius)); end = int(np.floor(c + radius + 1))
            k_start, k_end = 0, len(gaussian_kernel)
            if start < 0: k_start = -start; start = 0
            if end > seq_len: k_end = len(gaussian_kernel) - (end - seq_len); end = seq_len
            if start < end and k_start < k_end: weight_map[start:end] += gaussian_kernel[k_start:k_end]

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

def get_adaptive_threshold(signal, percentile, min_val):
    valid_vals = signal[signal > 0.001]
    if len(valid_vals) == 0: return min_val
    thresh = np.percentile(valid_vals, percentile)
    return max(min_val, thresh)

def heatmap_to_hitobjects(heatmap_np, timing_points, n_keys=4, frame_ms=10.0):
    # ... (保持原样生成 hit_objects_raw) ...
    total_frames = heatmap_np.shape[0]
    
    grid_weight = generate_grid_weight(timing_points, total_frames, frame_ms)
    weighted_heatmap = heatmap_np * grid_weight[:, None, None]
    
    tap_flat = weighted_heatmap[..., 0].flatten()
    tap_threshold = get_adaptive_threshold(tap_flat, HYPER_PARAMETERS['density_percentile'], HYPER_PARAMETERS['min_threshold'])
    hold_flat = heatmap_np[..., 1].flatten()
    hold_threshold = get_adaptive_threshold(hold_flat, 70, 0.01)
    
    print(f"  > Thresholds: Tap={tap_threshold:.4f}, Hold={hold_threshold:.4f}")
    
    hit_objects_raw = []
    column_width = int(512 / n_keys)
    min_dist = HYPER_PARAMETERS['min_dist']
    collision_gap = HYPER_PARAMETERS['collision_gap']
    
    for k in range(n_keys):
        tap_signal = weighted_heatmap[:, k, 0]
        hold_signal_raw = heatmap_np[:, k, 1]
        peaks, _ = find_peaks(tap_signal, height=tap_threshold, distance=min_dist)
        
        for i, p in enumerate(peaks):
            start_frame = p
            start_time = int(start_frame * frame_ms)
            x_pos = int((k + 0.5) * column_width)
            
            # 初步物理限制
            if i + 1 < len(peaks): max_duration = peaks[i+1] - start_frame - collision_gap
            else: max_duration = total_frames - start_frame - 1
            
            if max_duration <= 0:
                hit_objects_raw.append(f"{x_pos},192,{start_time},1,0,0:0:0:0:")
                continue

            # Hold 匹配
            check_window = hold_signal_raw[start_frame : start_frame + 5]
            is_hold = False
            if np.max(check_window) > hold_threshold:
                scan_len = min(max_duration, 500)
                hold_segment = hold_signal_raw[start_frame : start_frame + scan_len]
                cutoff = hold_threshold * 0.5
                below_cutoff = hold_segment < cutoff
                if np.any(below_cutoff): raw_duration = np.argmax(below_cutoff) 
                else: raw_duration = scan_len
                actual_duration = min(raw_duration, max_duration)
                
                if actual_duration >= 3:
                    is_hold = True
                    end_time = int((start_frame + actual_duration) * frame_ms)
                    hit_objects_raw.append(f"{x_pos},192,{start_time},128,0,{end_time}:0:0:0:0:")
            
            if not is_hold:
                hit_objects_raw.append(f"{x_pos},192,{start_time},1,0,0:0:0:0:")

    # [关键步骤] 调用强规则过滤器
    final_hit_objects = finalize_beatmap(hit_objects_raw, timing_points, n_keys)
    return final_hit_objects

def export_osu(meta_data, hit_objects, output_path):
    content = "osu file format v14\n\n"
    content += "[General]\n"
    content += f"AudioFilename: {meta_data.get('AudioFilename', 'audio.wav')}\n"
    content += f"AudioLeadIn: 0\n"
    content += "Mode: 3\n\n"
    content += "[Metadata]\n"
    content += f"Title:{meta_data.get('Title', 'Generated')}\n"
    content += f"TitleUnicode:{meta_data.get('TitleUnicode', 'Generated')}\n"
    content += f"Artist:{meta_data.get('Artist', 'BeatAnything')}\n"
    content += f"ArtistUnicode:{meta_data.get('ArtistUnicode', 'BeatAnything')}\n"
    content += f"Creator:BeatAnything_AI\n"
    content += f"Version:AI_Gen_V4\n\n"
    content += "[Difficulty]\n"
    content += f"HPDrainRate:{meta_data.get('Difficulty', {}).get('HPDrainRate', 8)}\n"
    content += f"CircleSize:{meta_data.get('Difficulty', {}).get('CircleSize', 4)}\n"
    content += f"OverallDifficulty:{meta_data.get('Difficulty', {}).get('OverallDifficulty', 8)}\n"
    content += "ApproachRate:5\n"
    content += "SliderMultiplier:1.4\n"
    content += "SliderTickRate:1\n\n"
    content += "[TimingPoints]\n"
    for tp in meta_data.get('TimingPoints', []):
        uninherited = 1 if tp['uninherited'] else 0
        line = f"{tp['time']},{tp['beatLength']},{tp['meter']},{tp['sampleSet']},{tp['sampleIndex']},{tp['volume']},{uninherited},{tp['effects']}"
        content += line + "\n"
    content += "\n"
    content += "[HitObjects]\n"
    for line in hit_objects:
        content += line + "\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully exported to: {output_path}")