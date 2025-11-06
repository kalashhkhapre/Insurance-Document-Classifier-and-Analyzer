from typing import Dict, List
from PIL import Image
from .utils import load_config
import os

class ImageAgent:
    """
    Visual reasoning on insurance document images.
    Decodes visual data from layouts, tables, stamps, signatures.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.temperature = self.config['agents']['image']['temperature']
        self.max_tokens = self.config['agents']['image']['max_tokens']
    
    def process(self, query: str, context: Dict, critical_fields: Dict) -> Dict:
        """
        Perform visual analysis on document images.
        
        Args:
            query: User query
            context: Unified context from General Agent
            critical_fields: Extracted fields from Critical Agent
            
        Returns:
            Visual analysis results
        """
        image_results = context['image_results']
        
        # Analyze relevant images
        visual_findings = self._analyze_images(image_results, critical_fields)
        
        # Detect visual elements
        visual_elements = self._detect_visual_elements(image_results)
        
        # Cross-validate with text
        validation = self._cross_validate_with_text(
            visual_findings, critical_fields
        )
        
        return {
            'visual_findings': visual_findings,
            'visual_elements': visual_elements,
            'validation': validation,
            'relevant_images': [r['image_path'] for r in image_results]
        }
    
    def _analyze_images(self, image_results: List[Dict], 
                       critical_fields: Dict) -> Dict:
        """Analyze images for visual information."""
        findings = {
            'has_tables': False,
            'has_stamps': False,
            'has_signatures': False,
            'has_logos': False,
            'layout_type': 'unknown'
        }
        
        # In production, this would use a vision model
        # For now, use heuristics based on image properties
        for result in image_results:
            image_path = result['image_path']
            
            if os.path.exists(image_path):
                img = Image.open(image_path)
                width, height = img.size
                
                # Simple heuristics
                aspect_ratio = width / height
                
                if 0.7 < aspect_ratio < 0.8:
                    findings['layout_type'] = 'standard_form'
                elif aspect_ratio > 1.2:
                    findings['has_tables'] = True
        
        return findings
    
    def _detect_visual_elements(self, image_results: List[Dict]) -> List[str]:
        """Detect specific visual elements in images."""
        elements = []
        
        # Check for common insurance document visual elements
        common_elements = [
            'Header section detected',
            'Table structure identified',
            'Signature field present',
            'Official stamp area visible',
            'Barcode/QR code detected'
        ]
        
        # In production, use actual vision model detection
        # For now, return placeholder
        for result in image_results[:2]:
            elements.extend(common_elements[:2])
        
        return list(set(elements))
    
    def _cross_validate_with_text(self, visual_findings: Dict, 
                                  critical_fields: Dict) -> Dict:
        """Cross-validate visual findings with extracted text."""
        validation = {
            'matched_fields': [],
            'discrepancies': []
        }
        
        # Check if visual layout supports extracted fields
        if visual_findings['has_tables'] and 'claim_amount' in critical_fields:
            validation['matched_fields'].append(
                'Claim amount likely from table structure'
            )
        
        if visual_findings['has_stamps'] and 'status' in critical_fields:
            validation['matched_fields'].append(
                'Status validated by stamp presence'
            )
        
        return validation
