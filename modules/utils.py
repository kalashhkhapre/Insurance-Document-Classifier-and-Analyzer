import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    return vec / np.linalg.norm(vec)

def ensure_dir(directory: str):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

class DocumentMetadata:
    """Store metadata for processed documents."""
    def __init__(self, doc_id: str, filename: str, pages: int):
        self.doc_id = doc_id
        self.filename = filename
        self.pages = pages
        self.page_metadata = []
    
    def add_page(self, page_id: int, text: str, image_path: str):
        self.page_metadata.append({
            "page_id": page_id,
            "text": text,
            "image_path": image_path
        })
    
    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "pages": self.pages,
            "page_metadata": self.page_metadata
        }
