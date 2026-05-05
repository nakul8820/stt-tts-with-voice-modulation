# download_sooktam2.py
# Run this once from your project root:
#   python download_sooktam2.py

from huggingface_hub import snapshot_download

print("Downloading Sooktam 2 model... (this may take a few minutes)")

snapshot_download(
    repo_id   = "bharatgenai/sooktam2",
    local_dir = "models/sooktam2",       # saved inside your project
)

print("\n✅ Done! Model saved to:  models/sooktam2/")
print("Next step: update MODEL_ID in providers/tts_sooktam2.py")
print('   Change:  MODEL_ID = "bharatgenai/sooktam2"')
print('   To:      MODEL_ID = "models/sooktam2"')
