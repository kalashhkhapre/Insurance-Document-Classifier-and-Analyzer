import re
from typing import Dict, List, Optional
from .utils import load_config

class CriticalAgent:
    """Enhanced Critical Agent with better invoice extraction."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        # Enhanced patterns for better extraction
        self.patterns = {
            # INVOICE PATTERNS (Enhanced)
            'invoice_number': [
                r'Invoice\s*(?:Number|No\.?|#)\s*:?\s*([A-Z0-9\-/]+)',
                r'Invoice\s*:?\s*([A-Z0-9\-/]{3,})',
                r'Bill\s*(?:No|Number)\s*:?\s*([A-Z0-9\-/]+)',
                r'Reference\s*(?:No|Number)\s*:?\s*([A-Z0-9\-/]+)',
                r'(?:INV|INVOICE)[\-\s]*([A-Z0-9]{3,})',
                r'\b([A-Z]{2,}\d{4,}|\d{4,}[A-Z]{2,})\b',  # Catches INV2024-001 or 2024INV001
                r'#\s*([A-Z0-9\-/]{3,})',
            ],
            'amount_due': [
                # Enhanced amount patterns
                r'(?:Total|Amount|Balance|Grand\s+Total|Amount\s+Due|Net\s+Amount)\s*:?\s*(?:₹|Rs\.?|INR|USD|\$)?\s*([\d,]+(?:\.\d{2})?)',
                r'(?:₹|Rs\.?|INR|USD|\$)\s*([\d,]+(?:\.\d{2})?)\s*(?:only|/-)?',
                r'Total\s*(?:Amount|Payable)?\s*:?\s*(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{2})?)',
                r'(?:Amount|Sum)\s*Payable\s*:?\s*(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{2})?)',
                r'(?:Grand|Final)\s*Total\s*:?\s*(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{2})?)',
                # Catches standalone amounts at end of lines
                r'(?:Total|Due|Payable).*?([\d,]{2,}(?:\.\d{2})?)\s*$',
                # Catches amounts in table format
                r'\|\s*Total\s*\|\s*(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{2})?)',
            ],
            'vendor_name': [
                r'(?:From|Vendor|Supplier|Company|Billed\s+by)\s*:?\s*([A-Z][A-Za-z\s&.,-]{3,50})',
                r'(?:Sold\s+by|Issued\s+by)\s*:?\s*([A-Z][A-Za-z\s&.,-]{3,50})',
                r'^([A-Z][A-Za-z\s&.,-]{5,50})(?:\n|$)',  # Company name at top
                r'(?:M/s|Messrs\.)\s+([A-Z][A-Za-z\s&.,-]{3,50})',
                # Catches "ABC Company Pvt Ltd" format
                r'([A-Z][A-Za-z\s]+(?:Pvt\.?\s+Ltd|Limited|LLC|Inc|Corp)\.?)',
            ],
            'description': [
                r'(?:Description|Details|Particulars|Service|Item)\s*:?\s*(.{10,200})',
                r'(?:For|Re:|Subject)\s*:?\s*(.{10,200})',
                r'Services\s+Rendered\s*:?\s*(.{10,200})',
                # Catches multi-line descriptions
                r'Description[:\s]+(.+?)(?:\n\s*\n|\n[A-Z])',
            ],
            'payment_terms': [
                r'(?:Payment\s+Terms?|Terms?|Due\s+Date|Payment\s+Due)\s*:?\s*(.{5,100})',
                r'(?:Net|Due\s+in)\s+(\d+\s+days?)',
                r'(?:Payable|Due)\s+(?:by|on|within)\s+(.{5,50})',
                r'(?:Terms?|Payment)\s*:?\s*(Net\s+\d+|COD|Advance|Credit)',
            ],
            'date': [
                r'Date\s*:?\s*(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})',
                r'(?:Invoice|Bill)\s+Date\s*:?\s*(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})',
                r'(?:Dated|On)\s*:?\s*(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})',
                r'(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})',
                r'(\d{4}[\-/]\d{1,2}[\-/]\d{1,2})',
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
            ],
            
            # POLICY PATTERNS
            'policy_number': [
                r'Policy\s*(?:Number|No\.?|#)\s*:?\s*([A-Z0-9\-/]+)',
                r'Policy\s*:?\s*([A-Z]{2,}\d{6,})',
                r'PN[:\-\s]*([A-Z0-9\-]+)',
                r'Policy\s+ID\s*:?\s*([A-Z0-9\-/]+)',
                r'\b([A-Z]{3,}[\-/]?\d{6,})\b',
            ],
            
            # CLAIM PATTERNS
            'claim_number': [
                r'Claim\s*(?:Number|No\.?|#)\s*:?\s*([A-Z0-9\-/]+)',
                r'Claim\s*:?\s*([A-Z]{2,}\d{6,})',
                r'CN[:\-\s]*([A-Z0-9\-]+)',
                r'Claim\s+ID\s*:?\s*([A-Z0-9\-/]+)',
            ],
            'claim_amount': [
                r'(?:Claim|Amount)\s*(?:Amount)?\s*:?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{2})?)',
                r'Amount\s*(?:Claimed|Due)\s*:?\s*(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{2})?)',
            ],
            'status': [
                r'Status\s*:?\s*(Approved|Pending|Under\s+Review|Rejected|Processing|Active|Settled)',
                r'\b(Approved|Pending|Rejected|Settled|Processing)\b',
            ],
            'insured_name': [
                r'(?:Insured|Policy\s+Holder|Name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
                r'(?:Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            ],
        }
    
    def process(self, context: Dict) -> Dict:
        """Extract critical fields from context with fallback strategies."""
        text_context = context['text_context']
        
        extracted_fields = {}
        confidence_scores = {}
        
        # Extract each field type
        for field_name, patterns in self.patterns.items():
            value, confidence = self._extract_field_with_fallback(
                text_context, 
                patterns,
                field_name
            )
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
    
    def _extract_field_with_fallback(self, text: str, patterns: List[str], 
                                     field_name: str) -> tuple:
        """Extract field with multiple fallback strategies."""
        
        # Strategy 1: Try all patterns
        for pattern_idx, pattern in enumerate(patterns):
            try:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Get the best match
                    best_match = max(matches, key=len) if matches else matches[0]
                    
                    if isinstance(best_match, tuple):
                        best_match = best_match[0]
                    
                    best_match = best_match.strip()
                    
                    # Validate match
                    if self._validate_match(best_match, field_name):
                        confidence = 0.95 if pattern_idx == 0 else max(0.7 - (pattern_idx * 0.05), 0.5)
                        
                        if len(best_match) >= 3:
                            confidence = min(confidence + 0.05, 0.99)
                        
                        return best_match, confidence
            except Exception as e:
                continue
        
        # Strategy 2: Fuzzy matching for specific fields
        if field_name in ['vendor_name', 'description']:
            fuzzy_result = self._fuzzy_extract(text, field_name)
            if fuzzy_result:
                return fuzzy_result, 0.60
        
        return None, 0.0
    
    def _validate_match(self, match: str, field_name: str) -> bool:
        """Validate extracted match based on field type."""
        if not match or len(match) < 2:
            return False
        
        # Amount validation
        if 'amount' in field_name.lower():
            # Should contain digits
            if not any(c.isdigit() for c in match):
                return False
            # Should not be just zeros
            if match.replace(',', '').replace('.', '').replace('0', '') == '':
                return False
        
        # Number validation (invoice, policy, claim)
        if 'number' in field_name.lower():
            # Should have at least 3 characters
            if len(match) < 3:
                return False
        
        # Name validation
        if 'name' in field_name.lower():
            # Should not be all numbers
            if match.replace(' ', '').isdigit():
                return False
        
        return True
    
    def _fuzzy_extract(self, text: str, field_name: str) -> Optional[str]:
        """Fallback fuzzy extraction for specific fields."""
        lines = text.split('\n')
        
        if field_name == 'vendor_name':
            # First non-empty line is often company name
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 5 and any(c.isupper() for c in line):
                    return line
        
        if field_name == 'description':
            # Look for longest line with mixed content
            candidates = [l.strip() for l in lines if len(l.strip()) > 10]
            if candidates:
                return max(candidates, key=len)[:200]
        
        return None
    
    def _identify_evidence_pages(self, context: Dict) -> List[int]:
        """Identify which pages contain critical information."""
        evidence_pages = set()
        
        for result in context.get('text_results', []):
            page_id = result['metadata']['page_id']
            if result['score'] > 0.4:
                evidence_pages.add(page_id)
        
        for result in context.get('image_results', []):
            page_id = result['metadata']['page_id']
            if result['score'] > 0.4:
                evidence_pages.add(page_id)
        
        return sorted(list(evidence_pages))
    
    def _generate_summary(self, fields: Dict) -> str:
        """Generate human-readable summary."""
        if not fields:
            return "No critical fields identified."
        
        summary_parts = []
        
        # Invoice fields
        if 'invoice_number' in fields:
            summary_parts.append(f"Invoice: {fields['invoice_number']}")
        if 'amount_due' in fields:
            summary_parts.append(f"Amount: ₹{fields['amount_due']}")
        if 'vendor_name' in fields:
            summary_parts.append(f"Vendor: {fields['vendor_name']}")
        
        # Policy fields
        if 'policy_number' in fields:
            summary_parts.append(f"Policy: {fields['policy_number']}")
        
        # Claim fields
        if 'claim_number' in fields:
            summary_parts.append(f"Claim: {fields['claim_number']}")
        if 'claim_amount' in fields:
            summary_parts.append(f"Amount: ₹{fields['claim_amount']}")
        
        # Common fields
        if 'date' in fields:
            summary_parts.append(f"Date: {fields['date']}")
        if 'status' in fields:
            summary_parts.append(f"Status: {fields['status']}")
        
        return " | ".join(summary_parts) if summary_parts else "Some fields extracted"
