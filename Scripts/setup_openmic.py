import os
import tarfile
from tqdm import tqdm
import pandas as pd
import json

try:
    import requests
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_URL = "https://zenodo.org/api/records/1432913/files/openmic-2018-v1.0.0.tgz/content"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "openmic-2018")
TGZ_PATH = os.path.join(BASE_DIR, "openmic-2018-v1.0.0.tgz")

INSTRUMENTS = [
    'accordion', 'bass', 'cello', 'clarinet', 'cymbals', 'drums', 'flute', 
    'guitar', 'mallet_percussion', 'mandolin', 'piano', 'saxophone', 
    'synthesizer', 'trombone', 'trumpet', 'ukulele', 'violin', 'voice'
]

# ==============================================================================
# 1. RESUMABLE DOWNLOAD — INFINITE RETRY WITH SLEEP
# ==============================================================================
def download_dataset(retry_wait=60):
    """
    Downloads with automatic resume support.
    On any connection failure (including DNS drops), waits `retry_wait` seconds
    then tries again — indefinitely — until the full file is on disk.
    Just leave this running; it will finish eventually.
    """
    import time

    if os.path.exists(DATA_DIR):
        print("Dataset folder already exists. Skipping download.")
        return

    # Resolve total file size once upfront
    def get_total_size():
        for _ in range(5):
            try:
                head = requests.head(DATASET_URL, allow_redirects=True, timeout=30)
                return int(head.headers.get("content-length", 0))
            except Exception:
                time.sleep(10)
        return 0

    total_size = get_total_size()
    attempt = 0

    while True:
        attempt += 1
        downloaded = os.path.getsize(TGZ_PATH) if os.path.exists(TGZ_PATH) else 0

        if total_size > 0 and downloaded >= total_size:
            print("Download already complete.")
            return

        pct = f"{downloaded / total_size * 100:.1f}%" if total_size else "unknown"
        print(f"\n[Attempt {attempt}] Resuming from {downloaded / 1e6:.1f} MB ({pct} of {total_size / 1e9:.2f} GB)...")

        headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
        mode    = "ab" if downloaded > 0 else "wb"

        try:
            with requests.get(DATASET_URL, stream=True, allow_redirects=True,
                              headers=headers, timeout=60) as r:
                r.raise_for_status()

                with open(TGZ_PATH, mode) as f, tqdm(
                    total=total_size,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading"
                ) as bar:
                    for chunk in r.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            print("\nDownload complete!")
            return

        except Exception as e:
            saved = os.path.getsize(TGZ_PATH) if os.path.exists(TGZ_PATH) else 0
            print(f"\n  Connection lost at {saved / 1e6:.1f} MB: {type(e).__name__}")
            print(f"  Waiting {retry_wait}s then retrying automatically...")
            time.sleep(retry_wait)

# ==============================================================================
# 2. EXTRACT DATASET
# ==============================================================================
def extract_dataset():
    if not os.path.exists(DATA_DIR):
        print("Extracting archive (this will take a few minutes)...")
        with tarfile.open(TGZ_PATH, 'r:gz') as tar:
            tar.extractall(path=BASE_DIR)
        print("Extraction complete!")
    else:
        print("Dataset folder already exists. Skipping extraction.")

# ==============================================================================
# 3. PREPARE MULTI-LABEL TARGETS
# ==============================================================================
def prepare_labels():
    print("Preparing multi-label mapping...")
    
    csv_path = os.path.join(DATA_DIR, "openmic-2018-aggregated-labels.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find {csv_path}. Extraction might have failed.")
        return
        
    df = pd.read_csv(csv_path)
    
    # OpenMIC is sparsely labeled. 
    # 'relevance' > 0.5 means the instrument is considered present.
    # We will build a dictionary: track_id -> {instrument: 1 or 0}
    
    labels_dict = {}
    
    # Initialize all known tracks with 0s for all instruments
    unique_tracks = df['sample_key'].unique()
    for track in unique_tracks:
        labels_dict[track] = {inst: 0 for inst in INSTRUMENTS}
        
    # Populate the 1s where relevance > 0.5
    for _, row in df.iterrows():
        track_id = row['sample_key']
        instrument = row['instrument']
        relevance = float(row['relevance'])
        
        # Only accept positive presence (and ensure it's in our target list)
        if relevance >= 0.5 and instrument in INSTRUMENTS:
            labels_dict[track_id][instrument] = 1

    # Save to a clean JSON file for our training script to use later
    output_json = os.path.join(DATA_DIR, "clean_multilabels.json")
    with open(output_json, "w") as f:
        json.dump(labels_dict, f, indent=4)
        
    print(f"Processed {len(unique_tracks)} audio tracks.")
    print(f"Labels saved to: {output_json}")
    print("\n[SUCCESS] Dataset is completely ready for training!")

if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    prepare_labels()
