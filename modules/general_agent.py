import numpy as np
from typing import Dict, List
from .utils import load_config, normalize_vector

class GeneralAgent:
    """
    Combines text and image retrieval results to form unified context.
    First agent in the pipeline - builds initial understanding.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.alpha = self.config['embeddings']['alpha']  # text weight
        self.beta = self.config['embeddings']['beta']    # image weight
    
    def process(self, query: str, text_results: List[Dict], 
                image_results: List[Dict]) -> Dict:
        """
        Fuse text and image retrieval results.
        
        Args:
            query: User query about insurance document
            text_results: Results from text retriever
            image_results: Results from image retriever
            
        Returns:
            Unified context dictionary
        """
        # Extract relevant text chunks
        text_context = "\n\n".join([
            f"[Text Chunk {i+1}, Score: {r['score']:.3f}]\n{r['text']}"
            for i, r in enumerate(text_results)
        ])
        
        # Extract relevant image information
        image_context = "\n".join([
            f"[Image Page {r['metadata']['page_id']}, Score: {r['score']:.3f}] "
            f"Path: {r['image_path']}"
            for r in image_results
        ])
        
        # Combine contexts
        unified_context = {
            'query': query,
            'text_context': text_context,
            'image_context': image_context,
            'text_results': text_results,
            'image_results': image_results,
            'fusion_weights': {
                'text_weight': self.alpha,
                'image_weight': self.beta
            }
        }
        
        # Generate initial interpretation
        interpretation = self._generate_interpretation(
            query, text_context, image_context
        )
        unified_context['interpretation'] = interpretation
        
        return unified_context
    
    def _generate_interpretation(self, query: str, text_ctx: str, 
                                 image_ctx: str) -> str:
        """Generate preliminary interpretation of the document."""
        prompt = f"""
        Query: {query}
        
        Available Text Context:
        {text_ctx[:1000]}  # Truncate for brevity
        
        Available Visual Context:
        {image_ctx}
        
        Preliminary Interpretation:
        Based on the retrieved context, this appears to be an insurance document containing:
        """
        
        # In production, this would call an LLM
        # For now, return a structured template
        return prompt
