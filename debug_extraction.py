#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.document_preprocessor import DocumentPreprocessor
from modules.critical_agent import CriticalAgent

def main():
    # Initialize
    preprocessor = DocumentPreprocessor()
    
    # UPDATE THIS PATH to your actual invoice file
    pdf_path = "data/raw_pdfs/invoice.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}")
        print("\nAvailable files in data/raw_pdfs/:")
        if os.path.exists("data/raw_pdfs/"):
            for file in os.listdir("data/raw_pdfs/"):
                print(f"  - {file}")
        else:
            print("  Directory doesn't exist!")
        return
    
    print(f"📄 Processing: {pdf_path}")
    print("="*60)
    
    # Process with higher DPI for better OCR
    doc_metadata = preprocessor.process_pdf(pdf_path, dpi=200)
    
    # Get all text
    full_text = " ".join([page['text'] for page in doc_metadata.page_metadata])
    
    print("\n" + "="*60)
    print("EXTRACTED TEXT (First 500 chars):")
    print("="*60)
    print(full_text[:500])
    print("\n")
    
    # Try extraction with Critical Agent
    agent = CriticalAgent()
    context = {
        'text_context': full_text, 
        'text_results': [], 
        'image_results': []
    }
    result = agent.process(context)
    
    print("="*60)
    print("EXTRACTED FIELDS:")
    print("="*60)
    
    if result['critical_fields']:
        for field, value in result['critical_fields'].items():
            confidence = result['confidence_scores'].get(field, 0)
            print(f"{field:20s}: {value} (Confidence: {confidence:.2%})")
    else:
        print("❌ No fields extracted!")
    
    print("\n")
    print("="*60)
    print("SUMMARY:")
    print("="*60)
    print(result['extraction_summary'])
    
    print("\n")
    print("="*60)
    print("FULL TEXT (for debugging):")
    print("="*60)
    print(full_text)

if __name__ == "__main__":
    main()
