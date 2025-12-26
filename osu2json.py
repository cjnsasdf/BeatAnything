# osu2json.py

import json
import os
import numpy as np
from PIL import Image
import imageio.v2 as imageio # 用于处理 float32 TIFF 图像

# ==============================================================================
# Part 1: .osu <-> .json Conversion
# ==============================================================================

def osu_to_json(osu_file_path):
    """
    将一个 osu!mania .osu 文件转换为详细的 .json 格式。
    """
    data = {"AudioFilename": "","Difficulty": {},"TimingPoints": [],"HitObjects": []}
    section, detected_mode = None, -1
    with open(osu_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"): continue
            if line.startswith('[') and line.endswith(']'): section = line[1:-1]; continue
            if section == "General":
                if line.startswith("AudioFilename:"): data["AudioFilename"] = line.split(":", 1)[1].strip()
                if line.startswith("Mode:"):
                    mode = int(line.split(":", 1)[1].strip())
                    if mode != 3: raise ValueError(f"谱面不是 mania 模式 (Mode: {mode})")
                    detected_mode = mode
            elif section == "Difficulty":
                key, value = line.split(":", 1); data["Difficulty"][key.strip()] = float(value.strip())
            elif section == "TimingPoints":
                parts = line.split(',');
                if len(parts) < 8: continue
                point = {"time": int(float(parts[0])), "beatLength": float(parts[1]), "meter": int(parts[2]),
                         "sampleSet": int(parts[3]), "sampleIndex": int(parts[4]), "volume": int(parts[5]),
                         "uninherited": int(parts[6]) == 1, "effects": int(parts[7])}
                data["TimingPoints"].append(point)
            elif section == "HitObjects":
                parts = line.split(',')
                time, obj_type = int(parts[2]), int(parts[3])
                column_count, x = int(data["Difficulty"]["CircleSize"]), int(parts[0])
                lane = int(x * column_count / 512)
                if obj_type & 128:
                    hit_object = {"time": time, "lane": lane, "type": "hold", "endTime": int(parts[5].split(':')[0])}
                else:
                    hit_object = {"time": time, "lane": lane, "type": "tap"}
                data["HitObjects"].append(hit_object)
    
    json_file_path = os.path.splitext(osu_file_path)[0] + ".json"
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return json_file_path

def json_to_osu(json_file_path, output_osu_path=None):
    """
    将详细的 .json 文件转换回 .osu 文件。
    """
    if output_osu_path is None:
        output_osu_path = os.path.splitext(json_file_path)[0] + "_restored.osu"
        
    with open(json_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    osu_content = "osu file format v14\n\n"
    osu_content += "[General]\nAudioFilename: {}\nMode: 3\n\n".format(data['AudioFilename'])
    osu_content += "[Editor]\nDistanceSpacing: 1\nBeatDivisor: 4\nGridSize: 4\nTimelineZoom: 1\n\n"
    osu_content += "[Metadata]\nTitle:Restored\nArtist:Unknown\nCreator:AI\nVersion:1.0\n\n"
    osu_content += "[Difficulty]\n"
    for key, value in data["Difficulty"].items(): osu_content += f"{key}:{value}\n"
    osu_content += "\n[Events]\n\n[TimingPoints]\n"
    for p in data["TimingPoints"]:
        line = f"{p['time']},{p['beatLength']},{p['meter']},{p['sampleSet']},{p['sampleIndex']},{p['volume']},{1 if p['uninherited'] else 0},{p['effects']}"
        osu_content += line + "\n"
    osu_content += "\n[HitObjects]\n"
    column_count = int(data["Difficulty"]["CircleSize"])
    for obj in data["HitObjects"]:
        x = int((obj["lane"] + 0.5) * 512 / column_count)
        if obj["type"] == "hold":
            osu_content += f"{x},192,{obj['time']},128,0,{obj['endTime']}:0:0:0:0:\n"
        else:
            osu_content += f"{x},192,{obj['time']},1,0,0:0:0:0:\n"
    with open(output_osu_path, 'w', encoding='utf-8') as f: f.write(osu_content)
    return output_osu_path

# ==============================================================================
# Part 2: .json <-> Tensors Conversion
# ==============================================================================

def json_to_tensors(json_file_path, output_dir, frame_resolution_ms=10):
    """
    将 .json 转换为张量 (float32 TIFF for timing, 2-channel PNG for hits)。
    """
    with open(json_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    n_keys = int(data['Difficulty']['CircleSize'])
    max_time = max([obj.get('endTime', obj['time']) for obj in data['HitObjects']] + [p['time'] for p in data['TimingPoints']] or [0])
    n_frames = (max_time // frame_resolution_ms) + 1

    tensor_a = np.zeros((n_frames, 7), dtype=np.float32)
    tensor_b = np.zeros((n_keys, n_frames, 2), dtype=np.uint8)

    if data['TimingPoints']:
        sorted_points = sorted(data['TimingPoints'], key=lambda x: x['time'])
        first_point, current_state = sorted_points[0], {}
        if first_point['uninherited']:
            current_state.update({'bpm_val': first_point['beatLength'], 'sv_val': 1.0, 'meter': float(first_point['meter'])})
        else:
            current_state.update({'bpm_val': 500.0, 'sv_val': -100.0 / first_point['beatLength'] if first_point['beatLength'] != 0 else 1.0, 'meter': 4.0})
        current_state.update({'volume': float(first_point['volume']), 'sample_set': float(first_point['sampleSet']), 'sample_idx': float(first_point['sampleIndex']), 'kiai': 1.0 if first_point['effects'] & 1 else 0.0})
        tensor_a[:, :] = np.array(list(current_state.values()), dtype=np.float32)
        for i, point in enumerate(sorted_points):
            start_frame = point['time'] // frame_resolution_ms
            end_frame = n_frames if i + 1 >= len(sorted_points) else sorted_points[i+1]['time'] // frame_resolution_ms
            if point['uninherited']:
                current_state.update({'bpm_val': point['beatLength'], 'meter': float(point['meter'])})
            else:
                current_state['sv_val'] = -100.0 / point['beatLength'] if point['beatLength'] != 0 else 1.0
            current_state.update({'volume': float(point['volume']), 'sample_set': float(point['sampleSet']), 'sample_idx': float(point['sampleIndex']), 'kiai': 1.0 if point['effects'] & 1 else 0.0})
            tensor_a[start_frame:end_frame, :] = np.array(list(current_state.values()), dtype=np.float32)
            
    for obj in data['HitObjects']:
        lane, start_frame = obj['lane'], obj['time'] // frame_resolution_ms
        if obj['type'] == 'tap' and start_frame < n_frames: tensor_b[lane, start_frame, 0] = 255
        elif obj['type'] == 'hold':
            end_frame = obj['endTime'] // frame_resolution_ms
            for f in range(start_frame, end_frame + 1):
                if f < n_frames: tensor_b[lane, f, 1] = 255
    
    os.makedirs(output_dir, exist_ok=True)
    imageio.imwrite(os.path.join(output_dir, "a_timing_states.tif"), tensor_a)
    for k in range(n_keys):
        for c, name in enumerate(['Tap', 'Hold']):
            Image.fromarray(tensor_b[k, :, c]).save(os.path.join(output_dir, f"b_key{k}_channel{c}_{name}.png"))

def tensors_to_json(input_dir, original_json_path, output_json_path, frame_resolution_ms=10):
    """
    从张量 (TIFF, PNGs) 恢复为 .json 文件。
    """
    with open(original_json_path, 'r', encoding='utf-8') as f: meta_data = json.load(f)
    tensor_a = imageio.imread(os.path.join(input_dir, "a_timing_states.tif"))
    n_keys = int(meta_data['Difficulty']['CircleSize'])
    n_frames = tensor_a.shape[0]
    tensor_b = np.zeros((n_keys, n_frames, 2), dtype=np.uint8)
    for k in range(n_keys):
        for c in range(2):
            fname = next(f for f in os.listdir(input_dir) if f.startswith(f"b_key{k}_channel{c}"))
            tensor_b[k, :, c] = np.array(Image.open(os.path.join(input_dir, fname))).squeeze()
            
    new_timing_points = []
    for f in range(n_frames):
        if f > 0 and np.array_equal(tensor_a[f], tensor_a[f-1]): continue
        state = tensor_a[f]; bpm_val, sv_mul, meter, volume, sample_set, sample_idx, kiai = state
        is_red_line = True
        if f > 0:
            last_state = tensor_a[f-1]
            if not np.isclose(last_state[0], state[0]) or not np.isclose(last_state[2], state[2]): is_red_line = True
            else: is_red_line = False
        if f == 0: is_red_line = True
        point = {"time": int(f * frame_resolution_ms), "beatLength": float(bpm_val) if is_red_line else float(np.round(-100.0 / sv_mul, 4)),
                 "meter": int(meter), "sampleSet": int(sample_set), "sampleIndex": int(sample_idx), "volume": int(volume),
                 "uninherited": bool(is_red_line), "effects": int(kiai)}
        new_timing_points.append(point)
        
    new_hit_objects = []
    for k in range(n_keys):
        is_holding, hold_start_time = False, 0
        for f in range(n_frames):
            if tensor_b[k, f, 0] > 128:
                new_hit_objects.append({"time": f * frame_resolution_ms, "lane": k, "type": "tap"})
            if tensor_b[k, f, 1] > 128 and not is_holding:
                is_holding = True; hold_start_time = f * frame_resolution_ms
            elif tensor_b[k, f, 1] < 128 and is_holding:
                is_holding = False
                if (f * frame_resolution_ms - hold_start_time) >= frame_resolution_ms:
                    new_hit_objects.append({"time": hold_start_time, "lane": k, "type": "hold", "endTime": (f - 1) * frame_resolution_ms})
        if is_holding:
            new_hit_objects.append({"time": hold_start_time, "lane": k, "type": "hold", "endTime": (n_frames - 1) * frame_resolution_ms})
            
    final_json = {"AudioFilename": meta_data["AudioFilename"], "Difficulty": meta_data["Difficulty"],
                  "TimingPoints": sorted(new_timing_points, key=lambda x: x['time']),
                  "HitObjects": sorted(new_hit_objects, key=lambda x: x['time'])}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=4)
    print(f"JSON successfully restored to {output_json_path}")
# --- Section 3: 如何使用 ---
if __name__ == '__main__':
    # !! 请务必修改为你自己的文件路径 !!
    OSU_FILE = "bj.HaLo - Ende (Loctav) [HD].osu"
    JSON_FILE = "mania_map.json"
    TENSOR_DIR = "./tensors/"
    RESTORED_JSON_FILE = "./mania_map_restored.json"

    # 1. 从 .osu 生成 .json
    try:
        json_file_path = osu_to_json(OSU_FILE)
        print(f".osu has been converted to {json_file_path}")

        # 2. 从 .json 生成位图张量
        json_to_tensors(json_file_path, TENSOR_DIR, frame_resolution_ms=10)

        # 3. 从位图张量恢复 .json
        tensors_to_json(TENSOR_DIR, json_file_path, RESTORED_JSON_FILE, frame_resolution_ms=10)

    except FileNotFoundError:
        print(f"Error: Input file not found. Please check the path: {OSU_FILE}")
    except Exception as e:
        print(f"An error occurred: {e}")