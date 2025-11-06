from typing import Dict
import json
from .utils import load_config, save_json
import os
from datetime import datetime

class SummarizerAgent:
    """
    Synthesizes all agent outputs into final structured result.
    Produces both JSON and human-readable summary.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.temperature = self.config['agents']['summarizer']['temperature']
        self.max_tokens = self.config['agents']['summarizer']['max_tokens']
    
    def process(self, query: str, general_context: Dict, critical_output: Dict,
                text_output: Dict, image_output: Dict) -> Dict:
        """
        Synthesize all agent outputs into final result.
        
        Args:
            query: Original user query
            general_context: Output from General Agent
            critical_output: Output from Critical Agent
            text_output: Output from Text Agent
            image_output: Output from Image Agent
            
        Returns:
            Final synthesized result with JSON and summary
        """
        # Build structured JSON output
        structured_data = self._build_structured_output(
            critical_output, text_output, image_output
        )
        
        # Generate human-readable summary
        summary = self._generate_summary(
            query, structured_data, critical_output, text_output, image_output
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            critical_output, text_output, image_output
        )
        
        final_result = {
            'query': query,
            'structured_data': structured_data,
            'summary': summary,
            'confidence_score': confidence,
            'evidence': {
                'pages': critical_output.get('evidence_pages', []),
                'images': image_output.get('relevant_images', []),
                'text_chunks': text_output.get('relevant_chunks', [])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return final_result
    
    def _build_structured_output(self, critical_output: Dict, 
                                 text_output: Dict, image_output: Dict) -> Dict:
        """Build structured JSON output."""
        critical_fields = critical_output.get('critical_fields', {})
        
        structured = {
            'Policy_Number': critical_fields.get('policy_number', 'N/A'),
            'Claim_Number': critical_fields.get('claim_number', 'N/A'),
            'Claim_Amount': critical_fields.get('claim_amount', 'N/A'),
            'Insured_Name': critical_fields.get('insured_name', 'N/A'),
            'Date': critical_fields.get('date', 'N/A'),
            'Status': critical_fields.get('status', 'N/A'),
            'Evidence_Pages': critical_output.get('evidence_pages', []),
            'Visual_Elements_Detected': image_output.get('visual_elements', []),
            'Consistency_Check': text_output.get('consistency', {}).get('is_consistent', True)
        }
        
        return structured
    
    def _generate_summary(self, query: str, structured_data: Dict,
                         critical_output: Dict, text_output: Dict, 
                         image_output: Dict) -> str:
        """Generate human-readable summary."""
        summary_parts = []
        
        # Opening statement
        policy_num = structured_data.get('Policy_Number', 'N/A')
        claim_num = structured_data.get('Claim_Number', 'N/A')
        
        if policy_num != 'N/A':
            summary_parts.append(
                f"Policy {policy_num} has been analyzed."
            )
        
        if claim_num != 'N/A':
            summary_parts.append(
                f"Claim {claim_num} was identified."
            )
        
        # Claim amount
        amount = structured_data.get('Claim_Amount', 'N/A')
        if amount != 'N/A':
            summary_parts.append(
                f"The claim amount is ₹{amount}."
            )
        
        # Status
        status = structured_data.get('Status', 'N/A')
        if status != 'N/A':
            summary_parts.append(
                f"Current status: {status}."
            )
        
        # Evidence
        evidence_pages = structured_data.get('Evidence_Pages', [])
        if evidence_pages:
            summary_parts.append(
                f"Evidence found on pages {', '.join(map(str, evidence_pages))}."
            )
        
        # Visual validation
        visual_elements = structured_data.get('Visual_Elements_Detected', [])
        if visual_elements:
            summary_parts.append(
                f"Visual analysis confirmed: {', '.join(visual_elements[:2])}."
            )
        
        # Consistency
        if not structured_data.get('Consistency_Check', True):
            summary_parts.append(
                "Note: Some inconsistencies were detected in the data."
            )
        
        return " ".join(summary_parts)
    
    def _calculate_confidence(self, critical_output: Dict, 
                             text_output: Dict, image_output: Dict) -> float:
        """Calculate overall confidence score."""
        scores = []
        
        # Critical field confidence
        confidence_scores = critical_output.get('confidence_scores', {})
        if confidence_scores:
            scores.extend(confidence_scores.values())
        
        # Text consistency
        consistency = text_output.get('consistency', {})
        if consistency.get('is_consistent', False):
            scores.append(0.9)
        else:
            scores.append(0.5)
        
        # Image validation
        validation = image_output.get('validation', {})
        if validation.get('matched_fields'):
            scores.append(0.85)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def save_result(self, result: Dict, output_dir: str = None):
        """Save result to file."""
        if output_dir is None:
            output_dir = self.config['paths']['results']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        save_json(result, filepath)
        
        return filepath
