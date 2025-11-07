import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import InsuranceDocumentAnalyzer
import json
from datetime import datetime
import pandas as pd
import uuid

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Insurance Document AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background-color: #f5f7fa;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
    
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .header-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .header-banner p {
        margin: 0.75rem 0 0 0;
        opacity: 0.95;
        font-size: 1.05rem;
        font-weight: 500;
    }
    
    .doc-item {
        background: white;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .doc-item:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    .doc-item.selected {
        background: #f0f4ff;
        border-color: #667eea;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    .field-display {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.2s;
    }
    
    .field-display:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    
    .field-label {
        color: #666;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.6rem;
        display: block;
    }
    
    .field-value {
        color: #222;
        font-size: 1.25rem;
        font-weight: 700;
        word-break: break-word;
        line-height: 1.5;
        margin: 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.45rem 1.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 0.5rem;
        letter-spacing: 0.3px;
    }
    
    .status-uploaded {
        background: #e3f2fd;
        color: #1565c0;
    }
    
    .status-classified {
        background: #f3e5f5;
        color: #6a1b9a;
    }
    
    .status-processed {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .status-queried {
        background: #fff3e0;
        color: #e65100;
    }
    
    .info-banner, .success-banner, .warning-banner, .error-banner {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        border-left: 5px solid;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .info-banner {
        background: #e3f2fd;
        border-color: #2196f3;
        color: #1565c0;
    }
    
    .success-banner {
        background: #e8f5e9;
        border-color: #4caf50;
        color: #2e7d32;
    }
    
    .warning-banner {
        background: #fff3e0;
        border-color: #ff9800;
        color: #e65100;
    }
    
    .error-banner {
        background: #ffebee;
        border-color: #f44336;
        color: #c62828;
    }
    
    .confidence-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .confidence-excellent {
        background: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    
    .confidence-good {
        background: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    
    .confidence-low {
        background: #ffebee;
        border-left: 5px solid #f44336;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DOCUMENT TYPE CONFIGS ====================
DOCUMENT_TYPES = {
    'Claim Form': {
        'icon': '📋',
        'color': '#4caf50',
        'fields': ['Claim Number', 'Policy Number', 'Claim Amount', 'Date', 'Insured Name', 'Status'],
    },
    'Invoice': {
        'icon': '💰',
        'color': '#ff9800',
        'fields': ['Invoice Number', 'Amount Due', 'Date', 'Vendor Name', 'Description', 'Payment Terms'],
    },
    'Inspection Report': {
        'icon': '🔍',
        'color': '#2196f3',
        'fields': ['Report Number', 'Inspection Date', 'Property', 'Damage Assessment', 'Recommendations', 'Inspector'],
    },
    'Policy Document': {
        'icon': '📄',
        'color': '#9c27b0',
        'fields': ['Policy Number', 'Coverage', 'Premium', 'Effective Date', 'Expiry Date', 'Terms'],
    },
    'Cover Letter': {
        'icon': '📮',
        'color': '#f44336',
        'fields': ['Date', 'Recipient', 'Subject', 'Attached Documents', 'Contact', 'Signature'],
    }
}

# ==================== SESSION STATE ====================
def initialize_session():
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'selected_doc_id' not in st.session_state:
        st.session_state.selected_doc_id = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

initialize_session()

# ==================== DOCUMENT MANAGER ====================
class DocumentManager:
    @staticmethod
    def add_document(filename, file_path, file_size):
        doc_id = str(uuid.uuid4())[:12]
        document = {
            'doc_id': doc_id,
            'filename': filename,
            'file_path': file_path,
            'file_size': file_size,
            'upload_time': datetime.now(),
            'status': 'Uploaded',
            'classification': None,
            'processed': False,
            'extraction_results': {},
            'queries': []
        }
        st.session_state.documents.append(document)
        return doc_id
    
    @staticmethod
    def get_document(doc_id):
        for doc in st.session_state.documents:
            if doc['doc_id'] == doc_id:
                return doc
        return None
    
    @staticmethod
    def update_document(doc_id, updates):
        for doc in st.session_state.documents:
            if doc['doc_id'] == doc_id:
                doc.update(updates)
                break
    
    @staticmethod
    def get_all_documents():
        return sorted(st.session_state.documents, key=lambda x: x['upload_time'], reverse=True)
    
    @staticmethod
    def delete_document(doc_id):
        st.session_state.documents = [d for d in st.session_state.documents if d['doc_id'] != doc_id]
    
    @staticmethod
    def get_status_summary():
        docs = st.session_state.documents
        return {
            'total': len(docs),
            'uploaded': sum(1 for d in docs if d['status'] == 'Uploaded'),
            'classified': sum(1 for d in docs if d['classification']),
            'processed': sum(1 for d in docs if d['processed']),
            'queried': sum(1 for d in docs if d['queries'])
        }

# ==================== INITIALIZE AI ====================
if st.session_state.analyzer is None:
    with st.spinner("🚀 Initializing AI System..."):
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config.yaml'
            )
            st.session_state.analyzer = InsuranceDocumentAnalyzer(config_path)
        except Exception as e:
            st.error(f"❌ Initialization Error: {e}")
            st.stop()

# ==================== MAIN HEADER ====================
st.markdown("""
<div class="header-banner">
    <h1>📄 Insurance Document AI Analyzer</h1>
    <p>Intelligent Classification • Data Extraction • Smart Querying</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR - DOCUMENT LIBRARY ====================
with st.sidebar:
    st.title("📚 Document Manager")
    
    # Upload section
    with st.expander("📤 Upload New Document", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose PDF",
            type=['pdf'],
            help="Upload insurance documents (PDF format only)"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"📄 {uploaded_file.name}")
            with col2:
                if st.button("✅", use_container_width=True):
                    pdf_path = f"data/raw_pdfs/{uploaded_file.name}"
                    os.makedirs("data/raw_pdfs", exist_ok=True)
                    
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    doc_id = DocumentManager.add_document(
                        uploaded_file.name,
                        pdf_path,
                        uploaded_file.size
                    )
                    st.session_state.selected_doc_id = doc_id
                    st.success("✅ Added to library!")
                    st.rerun()
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Statistics")
    summary = DocumentManager.get_status_summary()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", summary['total'])
        st.metric("Classified", summary['classified'])
    with col2:
        st.metric("Processed", summary['processed'])
        st.metric("Queried", summary['queried'])
    
    # Documents list
    st.markdown("---")
    st.subheader("📁 Your Documents")
    
    documents = DocumentManager.get_all_documents()
    
    if documents:
        for doc in documents:
            is_selected = doc['doc_id'] == st.session_state.selected_doc_id
            
            status_class = f"status-{doc['status'].lower()}"
            doc_type_icon = '❓'
            
            if doc['classification']:
                for dtype, config in DOCUMENT_TYPES.items():
                    if doc['classification']['document_type'] == dtype:
                        doc_type_icon = config['icon']
                        break
            
            btn_style = "primary" if is_selected else "secondary"
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button(
                    f"{doc_type_icon} {doc['filename'][:25]}...",
                    key=f"doc_{doc['doc_id']}",
                    use_container_width=True,
                    type=btn_style
                ):
                    st.session_state.selected_doc_id = doc['doc_id']
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{doc['doc_id']}", use_container_width=True):
                    DocumentManager.delete_document(doc['doc_id'])
                    if st.session_state.selected_doc_id == doc['doc_id']:
                        st.session_state.selected_doc_id = None
                    st.rerun()
    else:
        st.info("No documents yet")

# ==================== CHECK DOCUMENT SELECTED ====================
if not st.session_state.selected_doc_id:
    st.info("👈 Select or upload a document from the sidebar to begin")
    st.stop()

selected_doc = DocumentManager.get_document(st.session_state.selected_doc_id)

# ==================== DOCUMENT HEADER ====================
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    doc_type = selected_doc['classification']['document_type'] if selected_doc['classification'] else 'Unknown'
    icon = DOCUMENT_TYPES.get(doc_type, {}).get('icon', '❓')
    st.markdown(f"### {icon} {selected_doc['filename']}")

with col2:
    status = selected_doc['status']
    status_class = f"status-{status.lower()}"
    st.markdown(f"""<span class="status-badge {status_class}">{status}</span>""", unsafe_allow_html=True)

with col3:
    size_mb = selected_doc['file_size'] / 1024
    st.metric("Size", f"{size_mb:.1f} KB")

st.markdown("---")

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Classify", "💾 Process", "❓ Query", "📊 Results"])

# ==================== TAB 1: CLASSIFY ====================
with tab1:
    st.header("🔍 Document Classification")
    
    if selected_doc['classification']:
        result = selected_doc['classification']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Document Type", result['document_type'])
        
        with col2:
            st.metric("Confidence", f"{result['confidence_score']:.1%}")
        
        with col3:
            st.metric("Pages", result['pages'])
        
        st.markdown(f"""
        <div class="success-banner">
        <strong>📝 Description:</strong><br>{result['description']}
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📈 Classification Scores")
        
        probs = result['probabilities']
        prob_df = pd.DataFrame({
            'Document Type': list(probs.keys()),
            'Probability': list(probs.values())
        }).sort_values('Probability', ascending=False)
        
        st.bar_chart(prob_df.set_index('Document Type')['Probability'])
        
        if st.button("🔄 Re-classify", use_container_width=True):
            st.session_state.last_result = None
            selected_doc['classification'] = None
            selected_doc['status'] = 'Uploaded'
            st.rerun()
    
    else:
        st.markdown(f"""
        <div class="info-banner">
        Click the button below to classify this document
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Classify Document Now", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing document..."):
                try:
                    classification = st.session_state.analyzer.classify_document(selected_doc['file_path'])
                    
                    DocumentManager.update_document(selected_doc['doc_id'], {
                        'classification': classification,
                        'status': 'Classified'
                    })
                    
                    st.success("✅ Classification complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ==================== TAB 2: PROCESS ====================
with tab2:
    st.header("💾 Process & Index")
    
    if selected_doc['processed']:
        st.markdown(f"""
        <div class="success-banner">
        ✅ This document has been processed and indexed
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-banner">
        Process this document to enable querying
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚙️ Process Document", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📄 Extracting text...")
                progress_bar.progress(25)
                
                doc_metadata = st.session_state.analyzer.preprocessor.process_pdf(
                    selected_doc['file_path'], dpi=150
                )
                
                status_text.text("🔤 Creating text embeddings...")
                progress_bar.progress(50)
                
                st.session_state.analyzer.text_retriever.add_documents(doc_metadata.to_dict())
                
                status_text.text("🖼️ Creating image embeddings...")
                progress_bar.progress(75)
                
                st.session_state.analyzer.image_retriever.add_documents(doc_metadata.to_dict())
                
                status_text.text("💾 Saving indices...")
                progress_bar.progress(90)
                
                st.session_state.analyzer.text_retriever.save_index()
                st.session_state.analyzer.image_retriever.save_index()
                
                DocumentManager.update_document(selected_doc['doc_id'], {
                    'processed': True,
                    'status': 'Processed'
                })
                
                progress_bar.progress(100)
                status_text.empty()
                
                st.markdown(f"""
                <div class="success-banner">
                ✅ Successfully processed {doc_metadata.pages} pages!
                </div>
                """, unsafe_allow_html=True)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==================== TAB 3: QUERY ====================
with tab3:
    st.header("❓ Query Document")
    
    if not selected_doc['processed']:
        st.warning("⚠️ Please process this document first")
    else:
        try:
            if not st.session_state.analyzer.text_retriever.text_chunks:
                st.session_state.analyzer.load_indices()
        except:
            pass
        
        # Document type specific questions
        if selected_doc['classification']:
            doc_type = selected_doc['classification']['document_type']
            
            st.subheader("⚡ Suggested Questions")
            
            suggested_queries = {
                'Claim Form': [
                    "What is the claim number?",
                    "What is the claim amount?",
                    "What is the policy number?"
                ],
                'Invoice': [
                    "What is the invoice number?",
                    "What is the total amount due?",
                    "Who is the vendor?"
                ],
                'Inspection Report': [
                    "What damages were found?",
                    "Who conducted the inspection?",
                    "What are the recommendations?"
                ],
                'Policy Document': [
                    "What is the policy number?",
                    "What is the coverage?",
                    "What is the premium amount?"
                ],
                'Cover Letter': [
                    "What documents are attached?",
                    "Who is the recipient?",
                    "What is the date?"
                ]
            }
            
            queries = suggested_queries.get(doc_type, ["Ask a question 1", "Ask a question 2"])
            
            cols = st.columns(3)
            for idx, query in enumerate(queries):
                if cols[idx % 3].button(query, use_container_width=True):
                    st.session_state.query = query
        
        st.subheader("✍️ Custom Query")
        query = st.text_input(
            "Ask a question about this document:",
            value=st.session_state.get('query', ''),
            placeholder="What would you like to know?"
        )
        
        if st.button("🚀 Run Query", type="primary", use_container_width=True) and query:
            with st.spinner("🔍 Analyzing..."):
                try:
                    result = st.session_state.analyzer.query_document(query)
                    
                    # Store in session state
                    st.session_state.last_result = result
                    
                    # Also store in selected_doc for history
                    selected_doc['queries'].append({
                        'query': query,
                        'result': result,
                        'timestamp': datetime.now()
                    })
                    
                    # Update status
                    selected_doc['status'] = 'Queried'
                    
                    st.success("✅ Query complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")


# ==================== TAB 4: RESULTS ====================
with tab4:
    st.header("📊 Extraction Results")
    
    # Check if we have last_result in session
    if not st.session_state.last_result:
        st.info("Run a query first to see results")
    else:
        result = st.session_state.last_result
        structured = result.get('structured_data', {})
        
        doc_type = selected_doc['classification']['document_type'] if selected_doc['classification'] else 'Unknown'
        doc_config = DOCUMENT_TYPES.get(doc_type, {})
        
        st.subheader(f"🔍 Extracted Information from {doc_type}")
        
        # Field mapping for display
        field_mapping = {
            'invoice_number': 'Invoice_Number',
            'amount_due': 'Amount_Due',
            'vendor_name': 'Vendor_Name',
            'description': 'Description',
            'payment_terms': 'Payment_Terms',
            'date': 'Date',
            'policy_number': 'Policy_Number',
            'claim_number': 'Claim_Number',
            'claim_amount': 'Claim_Amount',
            'status': 'Status',
        }
        
        # Get fields from critical_fields OR structured_data
        critical_fields = result.get('critical_fields', {})
        
        # Combine both sources
        all_extracted_data = {**structured, **critical_fields}
        
        # Display fields based on document type
        fields_to_display = doc_config.get('fields', list(all_extracted_data.keys()))
        
        cols = st.columns(2)
        
        for idx, field in enumerate(fields_to_display):
            col = cols[idx % 2]
            
            with col:
                # Try to find the field in multiple places
                display_field = field.replace(' ', '_')
                
                # Check in critical_fields first
                if field.lower().replace(' ', '_') in critical_fields:
                    value = critical_fields[field.lower().replace(' ', '_')]
                    confidence = result.get('confidence_scores', {}).get(field.lower().replace(' ', '_'), 0.8)
                elif display_field in all_extracted_data:
                    value = all_extracted_data[display_field]
                    confidence = 0.8
                elif field in all_extracted_data:
                    value = all_extracted_data[field]
                    confidence = 0.8
                else:
                    value = "N/A"
                    confidence = 0.0
                
                # Format value
                if value and value != 'N/A':
                    value_display = str(value)
                    color = "#667eea"
                    icon = "-"
                else:
                    value_display = "Not Found"
                    color = "#ff6b6b"
                    icon = "-"
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 0.75rem 0; border-left: 5px solid {color};">
                <div style="color: #666; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.6rem;">{field}</div>
                <div style="color: #222; font-size: 1.25rem; font-weight: 700; word-wrap: break-word; line-height: 1.5;">{icon} {value_display}</div>
                <div style="color: #999; font-size: 0.75rem; margin-top: 0.5rem;">Confidence: {confidence:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Confidence score
        confidence = result.get('confidence_score', 0.5)
        confidence_pct = f"{confidence:.1%}"
        
        if confidence > 0.8:
            confidence_color = "#4caf50"
            confidence_status = "🟢 Excellent"
        elif confidence > 0.6:
            confidence_color = "#ff9800"
            confidence_status = "🟡 Good"
        else:
            confidence_color = "#f44336"
            confidence_status = "🔴 Low"
        
        st.markdown(f"""
        <div style="background: {confidence_color}15; padding: 1rem; border-radius: 8px; border-left: 4px solid {confidence_color}; margin: 1rem 0;">
        <strong style="color: {confidence_color};">🎯 Confidence Score: {confidence_pct} {confidence_status}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary
        st.markdown("---")
        st.subheader("📝 Analysis Summary")
        
        summary_text = result.get('summary', 'No summary available')
        st.markdown(f"""
        <div style="background: #e3f2fd; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2196f3; line-height: 1.7; font-size: 1rem; color: #1565c0;">
        {summary_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Export options
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Create export dataframe
            export_data = []
            for field in fields_to_display:
                if field.lower().replace(' ', '_') in critical_fields:
                    value = critical_fields[field.lower().replace(' ', '_')]
                elif field in all_extracted_data:
                    value = all_extracted_data[field]
                else:
                    value = 'N/A'
                
                export_data.append({'Field': field, 'Value': value})
            
            export_df = pd.DataFrame(export_data)
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 New Query", use_container_width=True, type="secondary"):
                st.session_state.last_result = None
                st.rerun()
        
        # Show raw result data
        with st.expander("🔧 View Raw Data (Debug)"):
            st.json(result)


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 2rem; font-size: 0.9rem;'>
<strong>Insurance Document AI v3.0</strong> | © 2025<br>
Document-Type Aware • Multi-Modal Analysis • Smart Extraction
</div>
""", unsafe_allow_html=True)
