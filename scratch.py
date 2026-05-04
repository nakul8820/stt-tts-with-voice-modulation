import torch
try:
    ckpt = torch.load("models/tts/mms-hin/mms_hin_full.pth", map_location="cpu")
    if isinstance(ckpt, dict):
        print("Keys:", list(ckpt.keys())[:10])
        if 'model' in ckpt:
            print("Model keys:", list(ckpt['model'].keys())[:10])
    else:
        print("Type:", type(ckpt))
except Exception as e:
    print("Error:", e)
