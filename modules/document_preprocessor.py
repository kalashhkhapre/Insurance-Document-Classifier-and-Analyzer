import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple
import uuid
import numpy as np
from .utils import load_config, ensure_dir, DocumentMetadata
import logging

logging.getLogger("ppocr").setLevel(logging.ERROR)

class DocumentPreprocessor:
    """Extract text and images from insurance PDFs - Optimized."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.ocr_engine = self.config['models']['ocr_engine']
        
        if self.ocr_engine == 'tesseract':
            try:
                pytesseract.get_tesseract_version()
            except pytesseract.TesseractNotFoundError:
                raise RuntimeError(
                    "Tesseract OCR is not installed. "
                    "Please install it using: sudo apt-get install tesseract-ocr"
                )
        
        ensure_dir(self.config['paths']['extracted_text'])
        ensure_dir(self.config['paths']['images'])
    
    def process_pdf(self, pdf_path: str, dpi: int = 200) -> DocumentMetadata:
        """
        Process PDF with optimized DPI for faster processing.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution (lower=faster, higher=better quality, default 150)
            
        Returns:
            DocumentMetadata object with processed information
        """
        doc_id = str(uuid.uuid4())
        filename = os.path.basename(pdf_path)
        
        print(f"Converting PDF to images (DPI: {dpi})...")
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=dpi, fmt='png')
        
        metadata = DocumentMetadata(doc_id, filename, len(pages))
        
        for page_num, page_image in enumerate(pages, start=1):
            # Progress update every 10 pages
            if page_num % 10 == 0:
                print(f"Processing page {page_num}/{len(pages)}...")
            
            # Save page image with compression
            image_path = os.path.join(
                self.config['paths']['images'],
                f"{doc_id}_page_{page_num}.png"
            )
            page_image.save(image_path, quality=85, optimize=True)
            
            # Extract text using OCR
            text = self._extract_text(page_image)
            
            # Save extracted text
            text_path = os.path.join(
                self.config['paths']['extracted_text'],
                f"{doc_id}_page_{page_num}.txt"
            )
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Add to metadata
            metadata.add_page(page_num, text, image_path)
        
        return metadata
    
    def _extract_text(self, image: Image) -> str:
        """Extract text from image using OCR."""
        if self.ocr_engine == 'tesseract':
            text = pytesseract.image_to_string(image, lang='eng')
        else:
            text = ""
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
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
