import re
from typing import Dict, List, Optional
from .utils import load_config

class CriticalAgent:
    """
    Extracts critical fields from insurance documents.
    Enhanced with better pattern matching.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        # Enhanced patterns for insurance document fields
        self.patterns = {
            'policy_number': [
                # Standard formats
                r'Policy\s*(?:Number|No\.?|#)\s*:?\s*([A-Z0-9\-/]+)',
                r'Policy\s*:?\s*([A-Z]{2,}\d{6,})',
                r'PN[:\-\s]*([A-Z0-9\-]+)',
                r'Policy\s+ID\s*:?\s*([A-Z0-9\-/]+)',
                # More flexible patterns
                r'(?:Policy|Pol\.)\s*(?:No|Number|#)?\s*[:.]?\s*([A-Z0-9]{6,})',
                r'([A-Z]{3,}[\-/]?\d{6,})',  # Catches patterns like ABC-123456
                r'\b([A-Z]{2}\d{8,})\b',  # Catches AB12345678
                r'Policy\s+([A-Z0-9\-/]{6,})',
            ],
            'claim_number': [
                r'Claim\s*(?:Number|No\.?|#)\s*:?\s*([A-Z0-9\-/]+)',
                r'Claim\s*:?\s*([A-Z]{2,}\d{6,})',
                r'CN[:\-\s]*([A-Z0-9\-]+)',
                r'Claim\s+ID\s*:?\s*([A-Z0-9\-/]+)',
                r'(?:Claim|Clm\.)\s*(?:No|Number|#)?\s*[:.]?\s*([A-Z0-9]{6,})',
            ],
            'claim_amount': [
                # Enhanced amount patterns
                r'(?:Claim|Amount|Total|Sum)\s*(?:Amount)?\s*:?\s*(?:₹|Rs\.?|INR|USD|\$)?\s*([\d,]+(?:\.\d{2})?)',
                r'(?:₹|Rs\.?|INR|USD|\$)\s*([\d,]+(?:\.\d{2})?)',
                r'Amount\s*(?:Claimed|Due|Payable)?\s*:?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{2})?)',
                r'Total\s*:?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{2})?)',
                r'(?:Payment|Payout)\s*:?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{2})?)',
                # More flexible
                r'\b([\d,]{4,}(?:\.\d{2})?)\b(?=\s*(?:rupees|INR|Rs|only))',
            ],
            'date': [
                # Various date formats
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
                r'(?:Date|Dated|On)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(?:Effective|Issue|Start)\s+Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            'status': [
                r'Status\s*:?\s*(Approved|Pending|Under\s+Review|Rejected|Processing|Active|Inactive|Expired|Settled)',
                r'(?:Claim|Policy)\s+Status\s*:?\s*(Approved|Pending|Under\s+Review|Rejected|Processing|Active|Inactive)',
                r'\b(Approved|Pending|Rejected|Settled|Processing)\b',
            ],
            'insured_name': [
                r'Insured\s*(?:Name|Person)?\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
                r'Policy\s*Holder\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
                r'Name\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
                r'(?:Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            ],
        }
    
    def process(self, context: Dict) -> Dict:
        """
        Extract critical fields from the unified context.
        
        Args:
            context: Unified context from General Agent
            
        Returns:
            Dictionary of extracted critical fields
        """
        text_context = context['text_context']
        
        extracted_fields = {}
        confidence_scores = {}
        
        # Extract each field type
        for field_name, patterns in self.patterns.items():
            value, confidence = self._extract_field(text_context, patterns)
            if value:
                extracted_fields[field_name] = value
                confidence_scores[field_name] = confidence
        
        # Identify evidence pages
        evidence_pages = self._identify_evidence_pages(context)
        
        return {
            'critical_fields': extracted_fields,
            'confidence_scores': confidence_scores,
            'evidence_pages': evidence_pages,
            'extraction_summary': self._generate_summary(extracted_fields)
        }
    
    def _extract_field(self, text: str, patterns: List[str]) -> tuple:
        """
        Extract field value using regex patterns with better matching.
        
        Returns:
            (value, confidence_score)
        """
        for pattern_idx, pattern in enumerate(patterns):
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Get the best match (longest or most specific)
                best_match = max(matches, key=len) if matches else matches[0]
                
                # Clean up the match
                if isinstance(best_match, tuple):
                    best_match = best_match[0]
                
                best_match = best_match.strip()
                
                # Confidence based on pattern priority and match quality
                confidence = 0.95 if pattern_idx == 0 else max(0.7 - (pattern_idx * 0.05), 0.5)
                
                # Boost confidence if match looks very specific
                if len(best_match) >= 8:  # Long matches are usually good
                    confidence = min(confidence + 0.05, 0.99)
                
                return best_match, confidence
        
        return None, 0.0
    
    def _identify_evidence_pages(self, context: Dict) -> List[int]:
        """Identify which pages contain critical information."""
        evidence_pages = set()
        
        # From text results
        for result in context.get('text_results', []):
            page_id = result['metadata']['page_id']
            if result['score'] > 0.5:  # Lower threshold
                evidence_pages.add(page_id)
        
        # From image results
        for result in context.get('image_results', []):
            page_id = result['metadata']['page_id']
            if result['score'] > 0.5:
                evidence_pages.add(page_id)
        
        return sorted(list(evidence_pages))
    
    def _generate_summary(self, fields: Dict) -> str:
        """Generate human-readable summary of extracted fields."""
        if not fields:
            return "No critical fields identified."
        
        summary_parts = []
        
        if 'policy_number' in fields:
            summary_parts.append(f"Policy Number: {fields['policy_number']}")
        
        if 'claim_number' in fields:
            summary_parts.append(f"Claim Number: {fields['claim_number']}")
        
        if 'claim_amount' in fields:
            summary_parts.append(f"Claim Amount: ₹{fields['claim_amount']}")
        
        if 'status' in fields:
            summary_parts.append(f"Status: {fields['status']}")
        
        if 'insured_name' in fields:
            summary_parts.append(f"Insured: {fields['insured_name']}")
        
        return " | ".join(summary_parts) if summary_parts else "Some fields extracted"
