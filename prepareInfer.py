import os
import argparse
import shutil
import json
import sys

# 尝试导入依赖
try:
    from pydub import AudioSegment
    import osu2json # 复用项目中的 osu2json.py
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请运行: pip install pydub")
    sys.exit(1)

# --- FFmpeg 配置 (与 create_dataset.py 保持一致) ---
FFMPEG_BIN_PATH = "./ffmpeg/bin" 
if os.path.isdir(FFMPEG_BIN_PATH) and os.path.exists(os.path.join(FFMPEG_BIN_PATH, "ffmpeg.exe")):
    os.environ["PATH"] += os.pathsep + os.path.abspath(FFMPEG_BIN_PATH)
else:
    # 检查系统路径
    if shutil.which("ffmpeg") is None:
        print("Warning: FFmpeg not found in ./ffmpeg/bin or PATH. Audio conversion might fail.")

def parse_audio_filename_from_osu(osu_path):
    """从 .osu 文件中读取 AudioFilename"""
    with open(osu_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().startswith("AudioFilename:"):
                return line.split(":", 1)[1].strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Prepare input files for BeatAnything inference.")
    parser.add_argument("osu_path", type=str, help="Path to the reference .osu file (provides BPM & Difficulty).")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to the audio file. If not set, will try to read from .osu.")
    parser.add_argument("--output_dir", type=str, default="./infer_data", help="Directory to save processed files.")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate (Default: 24000).")
    
    args = parser.parse_args()

    # 1. 检查输入
    osu_path = args.osu_path.strip('"').strip("'")
    if not os.path.exists(osu_path):
        print(f"Error: .osu file not found: {osu_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing: {os.path.basename(osu_path)}")
    print(f"Output Directory: {args.output_dir}")

    # ---------------------------------------------------------
    # 步骤 1: 转换 OSU -> JSON
    # ---------------------------------------------------------
    print("\n[1/2] Converting .osu to .json...")
    try:
        # 调用 osu2json 生成 json (默认生成在 osu 同级目录)
        generated_json_path = osu2json.osu_to_json(osu_path)
        
        # 移动到输出目录
        target_json_name = "beatmap_meta.json"
        target_json_path = os.path.join(args.output_dir, target_json_name)
        
        if os.path.exists(target_json_path):
            os.remove(target_json_path)
        shutil.move(generated_json_path, target_json_path)
        
        print(f"  -> Saved metadata to: {target_json_path}")
        
    except Exception as e:
        print(f"Error converting osu to json: {e}")
        return

    # ---------------------------------------------------------
    # 步骤 2: 处理音频 (Convert -> Mono 24k Wav)
    # ---------------------------------------------------------
    print("\n[2/2] Processing audio...")
    
    # 确定源音频路径
    src_audio_path = args.audio_path
    if src_audio_path:
        src_audio_path = src_audio_path.strip('"').strip("'")
    else:
        # 从 .osu 中读取音频文件名
        audio_filename = parse_audio_filename_from_osu(osu_path)
        if audio_filename:
            # 假设音频在 .osu 同级目录
            src_audio_path = os.path.join(os.path.dirname(osu_path), audio_filename)
        else:
            print("Error: Could not find 'AudioFilename' in .osu, and --audio_path was not provided.")
            return

    if not os.path.exists(src_audio_path):
        print(f"Error: Audio file not found at: {src_audio_path}")
        print("Please specify correct path using --audio_path")
        return

    target_audio_path = os.path.join(args.output_dir, "audio.wav")
    
    try:
        print(f"  -> Loading: {src_audio_path}")
        # 使用 Pydub 加载 (支持 mp3, ogg, wav 等)
        sound = AudioSegment.from_file(src_audio_path)
        
        print(f"  -> Converting to {args.sample_rate}Hz Mono...")
        # 强制转换为单声道和目标采样率
        sound = sound.set_channels(1).set_frame_rate(args.sample_rate)
        
        # 导出
        sound.export(target_audio_path, format="wav")
        print(f"  -> Saved audio to: {target_audio_path}")
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        print("Make sure ffmpeg is installed or configured correctly.")
        return

    # ---------------------------------------------------------
    # 完成
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Ready for Inference!")
    print(f"JSON:  {target_json_path}")
    print(f"Audio: {target_audio_path}")
    print("-" * 50)
    print("Run inference with:")
    print(f"python inference_full_song.py --json_path {target_json_path} --audio_path {target_audio_path} --vae_ckpt ./checkpoints_vae/vae_best.pth --reflow_ckpt ./checkpoints_reflow_gan/reflow_gan_best.pth --output_osu output.osu")
    print("="*50)

if __name__ == "__main__":
    main()