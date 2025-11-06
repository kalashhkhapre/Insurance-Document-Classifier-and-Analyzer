import os
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict
from .utils import load_config, save_json, load_json, ensure_dir

class ImageRetriever:
    """Image embedding and retrieval using CLIP (CPU only)."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        print(f"  Image Retriever using device: CPU")
        
        # Load CLIP model on CPU
        model_name = self.config['models']['image_encoder']
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embedding_dim = self.config['embeddings']['image_dim']
        
        # Use CPU-only FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.image_paths = []
        self.metadata = []
        
        ensure_dir(self.config['paths']['embeddings'])
    
    def add_documents(self, doc_metadata: Dict):
        """Add document page images to the index with batch processing."""
        images = []
        image_info = []
        
        # Load all images first
        for page in doc_metadata['page_metadata']:
            image_path = page['image_path']
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
                image_info.append((image_path, page))
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
        
        if not images:
            print("No images found to process")
            return
        
        # Batch process all images at once for efficiency
        print(f"  Processing {len(images)} images...")
        inputs = self.processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Normalize and add all at once
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings = image_features.cpu().numpy()
        
        # Add all embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        for i, (image_path, page) in enumerate(image_info):
            self.image_paths.append(image_path)
            self.metadata.append({
                'doc_id': doc_metadata['doc_id'],
                'page_id': page['page_id'],
                'image_path': image_path
            })
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant images using text query."""
        if top_k is None:
            top_k = self.config['retrieval']['top_k_image']
        
        # Ensure we have images to search
        if not self.image_paths:
            return []
        
        # Limit top_k to available images
        top_k = min(top_k, len(self.image_paths))
        
        # Encode text query
        query_embedding = self._encode_text(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.image_paths):
                results.append({
                    'image_path': self.image_paths[idx],
                    'metadata': self.metadata[idx],
                    'score': float(1 / (1 + distance))
                })
        
        return results
    
    def search_by_image(self, image: Image, top_k: int = None) -> List[Dict]:
        """Search for similar images."""
        if top_k is None:
            top_k = self.config['retrieval']['top_k_image']
        
        if not self.image_paths:
            return []
        
        top_k = min(top_k, len(self.image_paths))
        
        # Encode image
        query_embedding = self._encode_image(image)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.image_paths):
                results.append({
                    'image_path': self.image_paths[idx],
                    'metadata': self.metadata[idx],
                    'score': float(1 / (1 + distance))
                })
        
        return results
    
    def _encode_image(self, image: Image) -> np.ndarray:
        """Encode image to embedding vector."""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    
    def save_index(self, index_path: str = None):
        """Save FAISS index and metadata."""
        if index_path is None:
            index_path = os.path.join(
                self.config['paths']['embeddings'],
                'image_index.faiss'
            )
        
        faiss.write_index(self.index, index_path)
        
        metadata_path = index_path.replace('.faiss', '_metadata.json')
        save_json({
            'image_paths': self.image_paths,
            'metadata': self.metadata
        }, metadata_path)
        
        print(f"  ✓ Image index saved")
    
    def load_index(self, index_path: str = None):
        """Load FAISS index and metadata."""
        if index_path is None:
            index_path = os.path.join(
                self.config['paths']['embeddings'],
                'image_index.faiss'
            )
        
        self.index = faiss.read_index(index_path)
        
        metadata_path = index_path.replace('.faiss', '_metadata.json')
        data = load_json(metadata_path)
        self.image_paths = data['image_paths']
        self.metadata = data['metadata']
        
        print(f"  ✓ Image index loaded: {len(self.image_paths)} images")
