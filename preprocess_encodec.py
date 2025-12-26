# preprocess_encodec.py
import os
import torch
import torchaudio
import argparse
from tqdm import tqdm
from transformers import EncodecModel

def main():
    parser = argparse.ArgumentParser(description="Pre-compute EnCodec features.")
    parser.add_argument("data_dir", type=str, help="Path to dataset (e.g. data/train_sliced)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load EnCodec
    model_id = "facebook/encodec_24khz"
    print(f"Loading {model_id}...")
    try:
        model = EncodecModel.from_pretrained(model_id).to(device)
    except:
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id)
        model = EncodecModel.from_pretrained(model_dir).to(device)
    model.eval()

    TARGET_SR = 24000
    TARGET_LEN = 240000 # 10s

    subdirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    print(f"Found {len(subdirs)} samples. Starting extraction...")

    for sample_dir in tqdm(subdirs):
        audio_path = os.path.join(sample_dir, "audio.wav")
        output_path = os.path.join(sample_dir, "encodec_features.pt")

        if os.path.exists(output_path): continue
        if not os.path.exists(audio_path): continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            # Mix to mono
            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            # Resample
            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
            
            # Pad/Crop to exact 10s
            if waveform.shape[1] > TARGET_LEN:
                waveform = waveform[:, :TARGET_LEN]
            elif waveform.shape[1] < TARGET_LEN:
                waveform = torch.nn.functional.pad(waveform, (0, TARGET_LEN - waveform.shape[1]))
            
            # Extract
            waveform = waveform.unsqueeze(0).to(device) # [1, 1, T]
            with torch.no_grad():
                # Encoder output: [1, 128, 750]
                features = model.encoder(waveform)
            
            # Save as CPU tensor (Float32)
            torch.save(features.squeeze(0).cpu(), output_path)

        except Exception as e:
            print(f"Error {sample_dir}: {e}")

if __name__ == "__main__":
    main()