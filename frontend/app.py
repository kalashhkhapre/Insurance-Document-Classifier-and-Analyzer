import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import InsuranceDocumentAnalyzer
import json
from datetime import datetime
import pandas as pd
import uuid
from pathlib import Path

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Insurance Document AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .doc-item {
        background: white;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .doc-item:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .doc-item.selected {
        background: #f0f4ff;
        border-color: #667eea;
        border-left: 4px solid #667eea;
    }
    
    .field-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .field-label {
        color: #666;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .field-value {
        color: #222;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-uploaded {
        background: #e3f2fd;
        color: #1976d2;
    }
    
    .status-classified {
        background: #f3e5f5;
        color: #7b1fa2;
    }
    
    .status-processed {
        background: #e8f5e9;
        color: #388e3c;
    }
    
    .status-queried {
        background: #fff3e0;
        color: #f57c00;
    }
    
    .doc-type-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    
    .info-banner {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-banner {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-banner {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .error-banner {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .extraction-field {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DOCUMENT TYPE CONFIGS ====================
DOCUMENT_TYPES = {
    'Claim Form': {
        'icon': '📋',
        'color': '#4caf50',
        'fields': ['Claim Number', 'Policy Number', 'Claim Amount', 'Date', 'Insured Name', 'Status'],
        'patterns': {
            'Claim Number': 'claim.*number|clm.*no|claim.*id',
            'Policy Number': 'policy.*number|policy.*no|pol.*no',
            'Claim Amount': 'claim.*amount|amount.*claimed|total.*claim',
            'Date': 'date|filed|submitted',
            'Insured Name': 'insured|claimant|policy.*holder',
            'Status': 'status|approval|decision'
        }
    },
    'Invoice': {
        'icon': '💰',
        'color': '#ff9800',
        'fields': ['Invoice Number', 'Amount Due', 'Date', 'Vendor Name', 'Description', 'Payment Terms'],
        'patterns': {
            'Invoice Number': 'invoice.*number|invoice.*id|inv.*no',
            'Amount Due': 'amount.*due|total.*amount|subtotal|amount.*payable',
            'Date': 'invoice.*date|date|issued',
            'Vendor Name': 'vendor|supplier|from|bill.*from',
            'Description': 'description|service|item|details',
            'Payment Terms': 'payment.*terms|due.*date|net|payment'
        }
    },
    'Inspection Report': {
        'icon': '🔍',
        'color': '#2196f3',
        'fields': ['Report Number', 'Inspection Date', 'Property', 'Damage Assessment', 'Recommendations', 'Inspector'],
        'patterns': {
            'Report Number': 'report.*number|report.*id|reference',
            'Inspection Date': 'inspection.*date|date.*inspected|date',
            'Property': 'property|location|address',
            'Damage Assessment': 'damage|condition|assessment|findings',
            'Recommendations': 'recommendations|suggested|action|repair',
            'Inspector': 'inspector|surveyor|examined.*by'
        }
    },
    'Policy Document': {
        'icon': '📄',
        'color': '#9c27b0',
        'fields': ['Policy Number', 'Coverage', 'Premium', 'Effective Date', 'Expiry Date', 'Terms'],
        'patterns': {
            'Policy Number': 'policy.*number|policy.*no|pol.*no',
            'Coverage': 'coverage|covers|coverage.*limits',
            'Premium': 'premium|annual.*premium|payment',
            'Effective Date': 'effective.*date|starts|from.*date',
            'Expiry Date': 'expiry.*date|expires|end.*date',
            'Terms': 'terms|conditions|exclusions'
        }
    },
    'Cover Letter': {
        'icon': '📮',
        'color': '#f44336',
        'fields': ['Date', 'Recipient', 'Subject', 'Attached Documents', 'Contact', 'Signature'],
        'patterns': {
            'Date': 'date',
            'Recipient': 'to|dear|addressed',
            'Subject': 'subject|re:',
            'Attached Documents': 'attached|enclosed|submission',
            'Contact': 'contact|phone|email',
            'Signature': 'signature|signed|regards'
        }
    }
}

# ==================== SESSION STATE MANAGER ====================
def initialize_session():
    """Initialize all session state variables"""
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
    """Advanced document management system"""
    
    @staticmethod
    def add_document(filename, file_path, file_size):
        """Add document with unique tracking"""
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

# ==================== SIDEBAR ====================
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
            
            # Document item
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

# ==================== MAIN CONTENT ====================
if not st.session_state.selected_doc_id:
    st.info("👈 Select or upload a document from the sidebar to begin")
    st.stop()

selected_doc = DocumentManager.get_document(st.session_state.selected_doc_id)

# Document header
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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Classify", "💾 Process", "❓ Query", "📊 Results"])

# ==================== TAB 1: CLASSIFY ====================
with tab1:
    st.header("🔍 Document Classification")
    
    if selected_doc['classification']:
        result = selected_doc['classification']
        
        # Classification summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Document Type", result['document_type'])
        
        with col2:
            st.metric("Confidence", f"{result['confidence_score']:.1%}")
        
        with col3:
            st.metric("Pages", result['pages'])
        
        st.markdown(f"""
        <div class="success-banner">
        <strong>📝 Description:</strong> {result['description']}
        </div>
        """, unsafe_allow_html=True)
        
        # Probability distribution
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
        # Load indices
        try:
            if not st.session_state.analyzer.text_retriever.text_chunks:
                st.session_state.analyzer.load_indices()
        except:
            pass
        
        # Document type specific questions
        if selected_doc['classification']:
            doc_type = selected_doc['classification']['document_type']
            doc_config = DOCUMENT_TYPES.get(doc_type, {})
            
            st.subheader("⚡ Suggested Questions")
            
            # Create dynamic buttons based on document type
            cols = st.columns(3)
            suggested_queries = {
                'Claim Form': [
                    "What is the claim number?",
                    "What is the claim amount?",
                    "What is the policy number?"
                ],
                'Invoice': [
                    "What is the invoice number?",
                    "What is the total amount due?",
                    "What is the payment due date?"
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
            
            queries = suggested_queries.get(doc_type, ["Generic question 1", "Generic question 2", "Generic question 3"])
            
            for idx, query in enumerate(queries):
                if cols[idx].button(query, use_container_width=True):
                    st.session_state.query = query
        
        # Custom query
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
                    
                    # Save query
                    selected_doc['queries'].append({
                        'query': query,
                        'result': result,
                        'timestamp': datetime.now()
                    })
                    
                    st.session_state.last_result = result
                    selected_doc['status'] = 'Queried'
                    
                    st.success("✅ Query complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ==================== TAB 4: RESULTS ====================
with tab4:
    st.header("📊 Extraction Results")
    
    if not st.session_state.last_result:
        st.info("Run a query first to see results")
    else:
        result = st.session_state.last_result
        structured = result['structured_data']
        
        # Display extracted fields
        doc_type = selected_doc['classification']['document_type'] if selected_doc['classification'] else 'Unknown'
        doc_config = DOCUMENT_TYPES.get(doc_type, {})
        
        st.subheader(f"🔍 Extracted Information from {doc_type}")
        
        # Dynamic fields based on document type
        fields_to_display = doc_config.get('fields', [])
        field_mapping = {
            'Policy Number': 'Policy_Number',
            'Claim Number': 'Claim_Number',
            'Claim Amount': 'Claim_Amount',
            'Date': 'Date',
            'Status': 'Status',
            'Insured Name': 'Insured_Name',
            'Invoice Number': 'Invoice_Number',
            'Amount Due': 'Amount_Due',
            'Vendor Name': 'Vendor_Name',
            'Description': 'Description',
            'Payment Terms': 'Payment_Terms',
            'Property': 'Property',
            'Damage Assessment': 'Damage_Assessment',
            'Recommendations': 'Recommendations',
            'Inspector': 'Inspector',
            'Coverage': 'Coverage',
            'Premium': 'Premium',
            'Effective Date': 'Effective_Date',
            'Expiry Date': 'Expiry_Date',
            'Terms': 'Terms',
            'Recipient': 'Recipient',
            'Contact': 'Contact',
            'Signature': 'Signature'
        }
        
        # Display fields in columns
        cols = st.columns(2)
        
        for idx, field in enumerate(fields_to_display):
            col = cols[idx % 2]
            
            with col:
                db_field = field_mapping.get(field, field.replace(' ', '_'))
                value = structured.get(db_field, 'N/A')
                
                st.markdown(f"""
                <div class="extraction-field">
                <div class="field-label">{field}</div>
                <div class="field-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Summary and Confidence
        st.markdown("---")
        st.subheader("📝 Analysis Summary")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="info-banner">
            {result['summary']}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = result['confidence_score']
            color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            st.metric("Confidence", f"{confidence:.1%}", delta=f"{color}")
        
        # Export option
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(result, indent=2, default=str),
                file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 New Query", use_container_width=True):
                st.session_state.last_result = None
                st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 2rem;'>
<strong>Insurance Document AI v3.0</strong> | © 2025<br>
Document-Type Aware Classification • Smart Extraction • Intelligent Analysis
</div>
""", unsafe_allow_html=True)
