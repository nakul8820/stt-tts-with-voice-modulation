import torch

try:
    ckpt = torch.load("models/tts/mms-hin/mms_hin_full.pth", map_location="cpu")
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys())[:20])
        if 'model' in ckpt:
            print("Model keys:", list(ckpt['model'].keys())[:10])
        if 'state_dict' in ckpt:
            print("State dict keys:", list(ckpt['state_dict'].keys())[:10])
    else:
        print("Type:", type(ckpt))
except Exception as e:
    print("Error:", e)
