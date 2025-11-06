from huggingface_hub import hf_hub_download, snapshot_download
import os

def download_mdocagent_dataset():
    """Download MDocAgent dataset from HuggingFace."""
    
    # Download entire dataset
    print("Downloading MDocAgent dataset...")
    
    local_dir = "data/mdocagent_dataset"
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id="Lillianwei/Mdocagent-dataset",
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Dataset downloaded to {local_dir}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://huggingface.co/datasets/Lillianwei/Mdocagent-dataset")
        print("2. Download the files manually")
        print(f"3. Place them in: {local_dir}")

if __name__ == "__main__":
    download_mdocagent_dataset()
