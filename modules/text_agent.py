from typing import Dict, List
from .utils import load_config

class TextAgent:
    """
    Deep textual reasoning on insurance documents.
    Works on top-k text embeddings for extracting detailed answers.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.temperature = self.config['agents']['text']['temperature']
        self.max_tokens = self.config['agents']['text']['max_tokens']
    
    def process(self, query: str, context: Dict, critical_fields: Dict) -> Dict:
        """
        Perform deep textual analysis.
        
        Args:
            query: User query
            context: Unified context from General Agent
            critical_fields: Extracted fields from Critical Agent
            
        Returns:
            Detailed textual analysis results
        """
        text_results = context['text_results']
        
        # Analyze text chunks for detailed information
        detailed_analysis = self._analyze_text_chunks(
            query, text_results, critical_fields
        )
        
        # Extract relationships between entities
        relationships = self._extract_relationships(text_results, critical_fields)
        
        # Verify consistency
        consistency_check = self._check_consistency(critical_fields, text_results)
        
        return {
            'detailed_analysis': detailed_analysis,
            'relationships': relationships,
            'consistency': consistency_check,
            'relevant_chunks': [r['text'] for r in text_results[:3]]
        }
    
    def _analyze_text_chunks(self, query: str, text_results: List[Dict], 
                            critical_fields: Dict) -> str:
        """Analyze text chunks for detailed information."""
        analysis = []
        
        # Check for claim-related information
        for result in text_results[:3]:  # Top 3 results
            text = result['text'].lower()
            
            if 'claim' in query.lower():
                if any(word in text for word in ['damage', 'incident', 'accident']):
                    analysis.append(
                        f"Claim details found: {result['text'][:200]}..."
                    )
            
            if 'policy' in query.lower():
                if any(word in text for word in ['coverage', 'premium', 'term']):
                    analysis.append(
                        f"Policy information found: {result['text'][:200]}..."
                    )
        
        return "\n\n".join(analysis) if analysis else "No specific textual details found."
    
    def _extract_relationships(self, text_results: List[Dict], 
                              critical_fields: Dict) -> Dict:
        """Extract relationships between entities."""
        relationships = {}
        
        # Link policy to claim
        if 'policy_number' in critical_fields and 'claim_number' in critical_fields:
            relationships['policy_claim_link'] = {
                'policy': critical_fields['policy_number'],
                'claim': critical_fields['claim_number'],
                'relationship': 'associated'
            }
        
        # Link dates to events
        if 'date' in critical_fields and 'status' in critical_fields:
            relationships['timeline'] = {
                'date': critical_fields['date'],
                'status': critical_fields['status']
            }
        
        return relationships
    
    def _check_consistency(self, critical_fields: Dict, 
                          text_results: List[Dict]) -> Dict:
        """Check consistency of extracted information."""
        consistency = {
            'is_consistent': True,
            'issues': []
        }
        
        # Check if claim amount appears multiple times with same value
        if 'claim_amount' in critical_fields:
            amount = critical_fields['claim_amount']
            amount_count = sum(
                1 for r in text_results 
                if amount.replace(',', '') in r['text'].replace(',', '')
            )
            
            if amount_count == 0:
                consistency['is_consistent'] = False
                consistency['issues'].append(
                    "Claim amount not found in retrieved chunks"
                )
        
        return consistency
