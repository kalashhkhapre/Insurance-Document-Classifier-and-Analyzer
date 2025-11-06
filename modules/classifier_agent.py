import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import load_config

class DocumentClassifierAgent:
    """Classifies insurance documents based on content."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.model = SentenceTransformer(self.config['models']['text_encoder'])
        
        # Define document type characteristics
        self.document_types = {
            'Claim Form': {
                'keywords': [
                    'claim', 'claimant', 'claim number', 'incident', 'loss', 
                    'date of loss', 'description of loss', 'claim amount',
                    'fill in', 'form', 'signature', 'declaration'
                ],
                'patterns': ['claim.*form', 'claim.*submission', 'insurance.*claim'],
                'description': 'Document used to submit an insurance claim'
            },
            'Inspection Report': {
                'keywords': [
                    'inspection', 'report', 'surveyor', 'inspector', 'findings',
                    'damage', 'assessment', 'condition', 'photograph', 'recommendation',
                    'site inspection', 'physical condition', 'observations'
                ],
                'patterns': ['inspection.*report', 'surveyor.*report', 'damage.*assessment'],
                'description': 'Professional assessment of property/damage'
            },
            'Invoice': {
                'keywords': [
                    'invoice', 'billing', 'amount due', 'invoice number', 'date',
                    'bill', 'charges', 'payment', 'due date', 'description of services',
                    'qty', 'rate', 'total', 'from', 'to'
                ],
                'patterns': ['invoice.*number', 'bill.*to', 'amount.*due'],
                'description': 'Billing document for services/goods'
            },
            'Policy Document': {
                'keywords': [
                    'policy', 'coverage', 'premium', 'terms', 'conditions', 'exclusions',
                    'period', 'insured', 'deductible', 'coverage limits', 'effective date',
                    'policy holder', 'renewal', 'insurance'
                ],
                'patterns': ['insurance.*policy', 'policy.*document', 'coverage.*terms'],
                'description': 'Insurance policy terms and conditions'
            },
            'Cover Letter': {
                'keywords': [
                    'cover', 'letter', 'submission', 'enclosed', 'attached', 'please find',
                    'documents', 'regarding', 'reference', 'dear', 'sincerely', 'regards'
                ],
                'patterns': ['cover.*letter', 'submission.*letter', 'accompanying.*letter'],
                'description': 'Explanatory letter accompanying documents'
            }
        }
    
    def classify_document(self, text_chunks: List[str], image_count: int) -> Dict:
        """Classify document based on extracted text and images."""
        full_text = " ".join(text_chunks).lower()
        
        # Calculate scores for each document type
        classification_scores = {}
        
        for doc_type, characteristics in self.document_types.items():
            score = self._calculate_type_score(full_text, characteristics, text_chunks)
            classification_scores[doc_type] = score
        
        # Get top classification
        top_doc_type = max(classification_scores, key=classification_scores.get)
        confidence = classification_scores[top_doc_type]
        
        # Get probabilities
        total_score = sum(classification_scores.values())
        probabilities = {
            doc_type: (score / total_score) if total_score > 0 else 0.0
            for doc_type, score in classification_scores.items()
        }
        
        return {
            'document_type': top_doc_type,
            'confidence_score': min(confidence, 1.0),
            'all_scores': classification_scores,
            'probabilities': probabilities,
            'text_length': len(full_text),
            'image_count': image_count,
            'description': self.document_types[top_doc_type]['description']
        }
    
    def _calculate_type_score(self, text: str, characteristics: Dict, 
                             text_chunks: List[str]) -> float:
        """Calculate similarity score for a document type."""
        score = 0.0
        max_score = 0.0
        
        # Keyword matching (40% weight)
        keyword_score = self._keyword_matching_score(text, characteristics['keywords'])
        score += keyword_score * 0.4
        max_score += 0.4
        
        # Semantic similarity (40% weight)
        semantic_score = self._semantic_similarity_score(text_chunks, characteristics)
        score += semantic_score * 0.4
        max_score += 0.4
        
        # Pattern matching (20% weight)
        pattern_score = self._pattern_matching_score(text, characteristics['patterns'])
        score += pattern_score * 0.2
        max_score += 0.2
        
        return score / max_score if max_score > 0 else 0.0
    
    def _keyword_matching_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword matching score."""
        matched = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        return matched / len(keywords) if keywords else 0.0
    
    def _semantic_similarity_score(self, text_chunks: List[str], 
                                   characteristics: Dict) -> float:
        """Calculate semantic similarity using embeddings."""
        if not text_chunks:
            return 0.0
        
        keyword_text = " ".join(characteristics['keywords'])
        
        chunk_embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
        keyword_embedding = self.model.encode(keyword_text, convert_to_numpy=True)
        
        similarities = cosine_similarity([keyword_embedding], chunk_embeddings)[0]
        return float(np.mean(similarities)) if len(similarities) > 0 else 0.0
    
    def _pattern_matching_score(self, text: str, patterns: List[str]) -> float:
        """Calculate regex pattern matching score."""
        import re
        matched = 0
        
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                matched += 1
        
        return matched / len(patterns) if patterns else 0.0
    
    def get_classification_report(self, classification: Dict) -> str:
        """Generate human-readable classification report."""
        report = f"""
Document Classification Report
{'='*60}
Document Type: {classification['document_type']}
Confidence Score: {classification['confidence_score']:.1%}
Description: {classification['description']}

Probability Distribution:
{'-'*60}
"""
        sorted_probs = sorted(
            classification['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for doc_type, prob in sorted_probs:
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            report += f"{doc_type:20s} {prob:6.1%} {bar}\n"
        
        report += f"\nDocument Characteristics:\n"
        report += f"  Text Length: {classification['text_length']} characters\n"
        report += f"  Image Count: {classification['image_count']} images\n"
        
        return report
