import requests
import os
import sys
import time

model_url = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
save_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

print("Downloading hand_landmarker.task (7MB)...", flush=True)
print("This may take a few minutes...\n", flush=True)

try:
    response = requests.get(model_url, stream=True, timeout=300)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    start_time = time.time()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    mb_down = downloaded // 1024 // 1024
                    mb_total = total_size // 1024 // 1024
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = downloaded / elapsed / 1024 / 1024  # MB/s
                        print(f"\rProgress: {percent:5.1f}% ({mb_down}MB / {mb_total}MB) - Speed: {speed:.2f} MB/s", end='', flush=True)
    
    print()  # New line after progress
    file_size = os.path.getsize(save_path)
    print(f"✓ Download successful! ({file_size / 1024 / 1024:.1f} MB)", flush=True)
    
except KeyboardInterrupt:
    print("\n⚠ Download interrupted by user", flush=True)
    if os.path.exists(save_path):
        os.remove(save_path)
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}", flush=True)
    if os.path.exists(save_path):
        os.remove(save_path)
    sys.exit(1)



