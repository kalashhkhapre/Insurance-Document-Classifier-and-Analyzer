#!/usr/bin/env python3
import os
import sys

os.environ['FLAGS_log_level'] = '3'

import argparse
from modules.document_preprocessor import DocumentPreprocessor
from modules.text_retriever import TextRetriever
from modules.image_retriever import ImageRetriever
from modules.general_agent import GeneralAgent
from modules.critical_agent import CriticalAgent
from modules.text_agent import TextAgent
from modules.image_agent import ImageAgent
from modules.summarizer_agent import SummarizerAgent
from modules.classifier_agent import DocumentClassifierAgent
from modules.utils import load_config, save_json

class InsuranceDocumentAnalyzer:
    """Main pipeline for insurance document analysis and classification."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        print("Initializing components...")
        self.preprocessor = DocumentPreprocessor(config_path)
        self.text_retriever = TextRetriever(config_path)
        self.image_retriever = ImageRetriever(config_path)
        
        self.general_agent = GeneralAgent(config_path)
        self.critical_agent = CriticalAgent(config_path)
        self.text_agent = TextAgent(config_path)
        self.image_agent = ImageAgent(config_path)
        self.summarizer_agent = SummarizerAgent(config_path)
        self.classifier_agent = DocumentClassifierAgent(config_path)
        
        print("✓ Initialization complete!\n")
    
    def classify_document(self, pdf_path: str) -> dict:
        """Classify an insurance document."""
        print(f"\n{'='*60}")
        print(f"Classifying: {os.path.basename(pdf_path)}")
        print(f"{'='*60}\n")
        
        print("[1/3] Preprocessing document...")
        doc_metadata = self.preprocessor.process_pdf(pdf_path, dpi=150)
        print(f"✓ Extracted {doc_metadata.pages} pages")
        
        print("[2/3] Extracting text content...")
        text_chunks = []
        for page in doc_metadata.page_metadata:
            chunks = self.preprocessor.chunk_text(page['text'])
            text_chunks.extend(chunks)
        print(f"✓ Extracted {len(text_chunks)} text chunks")
        
        print("[3/3] Classifying document...")
        classification = self.classifier_agent.classify_document(
            text_chunks,
            doc_metadata.pages
        )
        print("✓ Classification complete")
        
        classification['filename'] = os.path.basename(pdf_path)
        classification['pages'] = doc_metadata.pages
        classification['doc_id'] = doc_metadata.doc_id
        
        result_path = os.path.join(
            self.config['paths']['results'],
            f"classification_{doc_metadata.doc_id}.json"
        )
        save_json(classification, result_path)
        
        return classification
    
    def process_document(self, pdf_path: str):
        """Process a single insurance document."""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(pdf_path)}")
        print(f"{'='*60}\n")
        
        print("[1/3] Preprocessing document...")
        doc_metadata = self.preprocessor.process_pdf(pdf_path, dpi=150)
        print(f"✓ Extracted {doc_metadata.pages} pages")
        
        print("[2/3] Generating embeddings...")
        self.text_retriever.add_documents(doc_metadata.to_dict())
        self.image_retriever.add_documents(doc_metadata.to_dict())
        print("✓ Embeddings generated and indexed")
        
        print("[3/3] Saving indices...")
        self.text_retriever.save_index()
        self.image_retriever.save_index()
        print("✓ Indices saved")
        
        metadata_path = os.path.join(
            self.config['paths']['results'],
            f"{doc_metadata.doc_id}_metadata.json"
        )
        save_json(doc_metadata.to_dict(), metadata_path)
        
        return doc_metadata
    
    def load_indices(self):
        """Load pre-built indices for querying."""
        print("Loading indices...")
        try:
            self.text_retriever.load_index()
            self.image_retriever.load_index()
            return True
        except Exception as e:
            print(f"✗ Failed to load indices: {e}")
            print("\nNo indices found. Please process a document first:")
            print("  python main.py --mode process --pdf data/raw_pdfs/your_document.pdf")
            return False
    
    def query_document(self, query: str) -> dict:
        """Query the processed documents."""
        if not self.text_retriever.text_chunks or not self.image_retriever.image_paths:
            if not self.load_indices():
                return None
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        print("[1/5] General Agent - Retrieving context...")
        text_results = self.text_retriever.search(query)
        image_results = self.image_retriever.search(query)
        general_context = self.general_agent.process(query, text_results, image_results)
        print(f"✓ Retrieved {len(text_results)} text chunks and {len(image_results)} images")
        
        print("[2/5] Critical Agent - Extracting fields...")
        critical_output = self.critical_agent.process(general_context)
        print(f"✓ Extracted {len(critical_output['critical_fields'])} critical fields")
        
        print("[3/5] Text Agent - Analyzing text...")
        text_output = self.text_agent.process(
            query, general_context, critical_output['critical_fields']
        )
        print("✓ Textual analysis complete")
        
        print("[4/5] Image Agent - Analyzing visuals...")
        image_output = self.image_agent.process(
            query, general_context, critical_output['critical_fields']
        )
        print("✓ Visual analysis complete")
        
        print("[5/5] Summarizer Agent - Synthesizing results...")
        final_result = self.summarizer_agent.process(
            query, general_context, critical_output, text_output, image_output
        )
        print("✓ Analysis complete")
        
        result_path = self.summarizer_agent.save_result(final_result)
        print(f"\n✓ Results saved to: {result_path}")
        
        # Ensure the result has all necessary keys for Streamlit UI
        if not isinstance(final_result, dict):
            final_result = {}
        
        # Add missing keys if needed
        if 'structured_data' not in final_result:
            final_result['structured_data'] = critical_output.get('critical_fields', {})
        
        if 'critical_fields' not in final_result:
            final_result['critical_fields'] = critical_output.get('critical_fields', {})
        
        if 'confidence_scores' not in final_result:
            final_result['confidence_scores'] = critical_output.get('confidence_scores', {})
        
        if 'confidence_score' not in final_result:
            scores = critical_output.get('confidence_scores', {})
            if scores:
                final_result['confidence_score'] = sum(scores.values()) / len(scores)
            else:
                final_result['confidence_score'] = 0.0
        
        if 'summary' not in final_result:
            final_result['summary'] = critical_output.get('extraction_summary', 'Analysis complete')
        
        if 'evidence_pages' not in final_result:
            final_result['evidence_pages'] = critical_output.get('evidence_pages', [])
        
        return final_result

    
    def print_classification(self, classification: dict):
        """Print classification results."""
        if classification is None:
            return
        
        print(self.classifier_agent.get_classification_report(classification))
        
        print(f"\n📎 Document Details:")
        print(f"  Filename: {classification['filename']}")
        print(f"  Pages: {classification['pages']}")
        print(f"  Document ID: {classification['doc_id']}")
    
    def print_result(self, result: dict):
        """Pretty print the analysis result."""
        if result is None:
            return
        
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}\n")
        
        print("📊 Structured Data:")
        print("-" * 60)
        structured = result['structured_data']
        for key, value in structured.items():
            if key != 'Visual_Elements_Detected' and key != 'Evidence_Pages':
                print(f"  {key:20s}: {value}")
        
        print(f"\n  Evidence Pages: {structured.get('Evidence_Pages', [])}")
        
        print(f"\n📝 Summary:")
        print("-" * 60)
        print(f"  {result['summary']}")
        
        print(f"\n🎯 Confidence Score: {result['confidence_score']:.2%}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Insurance Document Analyzer - Extract & Classify Insurance PDFs"
    )
    parser.add_argument(
        '--mode', 
        choices=['process', 'query', 'classify', 'both'],
        required=True,
        help='Mode: process (index), query (ask questions), classify (document type), or both'
    )
    parser.add_argument('--pdf', help='Path to PDF file to process')
    parser.add_argument('--query', help='Question to ask about the documents')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    analyzer = InsuranceDocumentAnalyzer(args.config)
    
    if args.mode == 'classify':
        if not args.pdf:
            print("Error: --pdf required for classify mode")
            return
        classification = analyzer.classify_document(args.pdf)
        analyzer.print_classification(classification)
        return
    
    if args.mode in ['process', 'both']:
        if not args.pdf:
            print("Error: --pdf required for process mode")
            return
        analyzer.process_document(args.pdf)
    
    if args.mode in ['query', 'both']:
        if not args.query:
            print("Error: --query required for query mode")
            return
        result = analyzer.query_document(args.query)
        analyzer.print_result(result)


if __name__ == "__main__":
    main()
