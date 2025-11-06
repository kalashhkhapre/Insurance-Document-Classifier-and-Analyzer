import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .utils import load_config, save_json, load_json, ensure_dir

class TextRetriever:
    """Text embedding and retrieval using sentence transformers (CPU only)."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        print(f"  Text Retriever using device: CPU")
        
        # Load model on CPU
        self.model = SentenceTransformer(
            self.config['models']['text_encoder'],
            device='cpu'
        )
        self.embedding_dim = self.config['embeddings']['text_dim']
        
        # Use CPU-only FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.text_chunks = []
        self.metadata = []
        
        ensure_dir(self.config['paths']['embeddings'])
    
    def add_documents(self, doc_metadata: Dict):
        """Add document chunks to the index with batch encoding."""
        all_chunks = []
        all_metadata = []
        
        # Collect all chunks from all pages
        for page in doc_metadata['page_metadata']:
            text = page['text']
            chunks = self._chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'doc_id': doc_metadata['doc_id'],
                    'page_id': page['page_id'],
                    'chunk_id': chunk_idx,
                    'image_path': page['image_path']
                })
        
        if not all_chunks:
            print("No text chunks found to process")
            return
        
        # Batch encode all chunks at once
        print(f"  Processing {len(all_chunks)} text chunks...")
        embeddings = self.model.encode(
            all_chunks,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False
        )
        
        # Add all embeddings to index at once
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and metadata
        self.text_chunks.extend(all_chunks)
        self.metadata.extend(all_metadata)
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant text chunks."""
        if top_k is None:
            top_k = self.config['retrieval']['top_k_text']
        
        if not self.text_chunks:
            return []
        
        top_k = min(top_k, len(self.text_chunks))
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.text_chunks):
                results.append({
                    'text': self.text_chunks[idx],
                    'metadata': self.metadata[idx],
                    'score': float(1 / (1 + distance))
                })
        
        return results
    
    def save_index(self, index_path: str = None):
        """Save FAISS index and metadata."""
        if index_path is None:
            index_path = os.path.join(
                self.config['paths']['embeddings'],
                'text_index.faiss'
            )
        
        faiss.write_index(self.index, index_path)
        
        metadata_path = index_path.replace('.faiss', '_metadata.json')
        save_json({
            'text_chunks': self.text_chunks,
            'metadata': self.metadata
        }, metadata_path)
        
        print(f"  ✓ Text index saved")
    
    def load_index(self, index_path: str = None):
        """Load FAISS index and metadata."""
        if index_path is None:
            index_path = os.path.join(
                self.config['paths']['embeddings'],
                'text_index.faiss'
            )
        
        self.index = faiss.read_index(index_path)
        
        metadata_path = index_path.replace('.faiss', '_metadata.json')
        data = load_json(metadata_path)
        self.text_chunks = data['text_chunks']
        self.metadata = data['metadata']
        
        print(f"  ✓ Text index loaded: {len(self.text_chunks)} chunks")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = self.config['embeddings']['chunk_size']
        overlap = self.config['embeddings']['chunk_overlap']
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
