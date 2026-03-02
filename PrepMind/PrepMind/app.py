import streamlit as st
import os
import tempfile
import time
from ingestion import extract_text_from_pdf, extract_text_from_url, chunk_text
from vector_store import add_to_vector_store
from llm_pipeline import generate_rag_response, generate_pure_llm_response
from logging_utils import log_ingestion_metrics, get_metrics_summary

SUBJECTS = ["DSA", "DBMS", "OS", "Computer Networks", "System Design", "General Programming"]

st.set_page_config(page_title="PrepMind - AI Study Assistant", layout="wide", page_icon="🧠")

st.title("🧠 PrepMind: AI Exam & Placement Assistant")
st.markdown("Your personal study AI. Select a subject, add your materials, and ask questions.")

# --------- SIDEBAR UI ---------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_subject = st.selectbox("Select Subject", SUBJECTS)
    
    exam_mode = st.radio("Select Mode", ["Exam", "Placement"])
    
    st.divider()
    
    st.header("🗂️ Add Knowledge Base")
    st.write("Upload PDFs or provide URLs to enhance the RAG system.")
    
    uploaded_file = st.file_uploader("Upload PDF Reference", type=["pdf"])
    
    st.caption("OR")
    
    reference_url = st.text_input("Enter Website URL (e.g., geeksforgeeks)")
    
    if st.button("Ingest Knowledge"):
        with st.spinner("Processing documents..."):
            start_time = time.time()
            text = ""
            source_name = "Unknown"
            source_type = "Unknown"
            
            try:
                if uploaded_file is not None:
                    text = extract_text_from_pdf(uploaded_file)
                    source_name = uploaded_file.name
                    source_type = "PDF"
                    
                elif reference_url:
                    text = extract_text_from_url(reference_url)
                    source_name = reference_url
                    source_type = "URL"
                    
                if text:
                    chunks = chunk_text(text)
                    ingestion_time = time.time() - start_time
                    
                    if chunks:
                        success = add_to_vector_store(selected_subject, chunks, source_name)
                        if success:
                            st.success(f"Successfully added {len(chunks)} chunks to {selected_subject} knowledge base!")
                            log_ingestion_metrics(
                                source=source_name,
                                source_type=source_type,
                                num_chunks=len(chunks),
                                success=True
                            )
                        else:
                            st.error("Failed to add to database.")
                            log_ingestion_metrics(
                                source=source_name,
                                source_type=source_type,
                                num_chunks=0,
                                success=False,
                                error="Failed to add to vector store"
                            )
                    else:
                        st.warning("No chunks created from text.")
                        log_ingestion_metrics(
                            source=source_name,
                            source_type=source_type,
                            num_chunks=0,
                            success=False,
                            error="No chunks created"
                        )
                else:
                    st.warning("No text extracted. Please provide a valid PDF or URL.")
                    log_ingestion_metrics(
                        source=source_name if source_name != "Unknown" else (reference_url if reference_url else "Unknown"),
                        source_type=source_type,
                        num_chunks=0,
                        success=False,
                        error="No text extracted"
                    )
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
                log_ingestion_metrics(
                    source=source_name if source_name != "Unknown" else (reference_url if reference_url else "Unknown"),
                    source_type=source_type,
                    num_chunks=0,
                    success=False,
                    error=str(e)
                )
                
    st.divider()
    comparison_mode = st.toggle("🔍 Enable Comparison Mode", value=False, help="Compare Pure LLM vs RAG Response")
    
    st.divider()
    with st.expander("📊 System Metrics"):
        metrics_info = get_metrics_summary()
        st.caption(f"Logging: {metrics_info['logging_enabled']}")
        st.caption(f"Log Directory: {metrics_info['log_directory']}")


# --------- MAIN INTERFACE ---------

st.subheader("Got a Question?")
query = st.text_area("Ask anything about the subject...", height=100)

if st.button("Generate Explanation", type="primary"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Generating Response..."):
            try:
                if comparison_mode:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🤖 Pure LLM Response")
                        pure_response = generate_pure_llm_response(query, exam_mode)
                        st.markdown(pure_response)
                        
                    with col2:
                        st.subheader("📚 RAG Enhanced Response")
                        rag_result = generate_rag_response(selected_subject, query, exam_mode)
                        st.markdown(rag_result["answer"])
                        
                        # Display metrics
                        if "metrics" in rag_result:
                            with st.expander("📊 Query Metrics"):
                                st.metric("Response Time", f"{rag_result['metrics']['response_time']:.2f}s")
                                st.metric("Tokens Used (est.)", rag_result['metrics']['num_tokens'])
                                st.metric("Sources Retrieved", rag_result['metrics']['sources_retrieved'])
                                st.metric("Avg Similarity Score", f"{rag_result['metrics']['avg_similarity']:.4f}")
                        
                        # Similarity Sources
                        st.divider()
                        st.markdown("#### Retrieved Sources")
                        for i, source in enumerate(rag_result["sources"], 1):
                            st.caption(f"**{i}. Source:** {source['source']} (Score: {source['score']:.4f})")
                else:
                    # RAG Only
                    rag_result = generate_rag_response(selected_subject, query, exam_mode)
                    st.markdown(rag_result["answer"])
                    
                    # Display metrics
                    if "metrics" in rag_result:
                        with st.expander("📊 Query Metrics"):
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Response Time", f"{rag_result['metrics']['response_time']:.2f}s")
                            col2.metric("Tokens (est.)", rag_result['metrics']['num_tokens'])
                            col3.metric("Sources", rag_result['metrics']['sources_retrieved'])
                            col4.metric("Avg Similarity", f"{rag_result['metrics']['avg_similarity']:.4f}")
                    
                    with st.expander("View Retrieved Context & Scores"):
                        for i, source in enumerate(rag_result["sources"], 1):
                            st.write(f"**Source {i}:** {source['source']}")
                            st.write(f"*Similarity Score:* {source['score']:.4f}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Have you configured your API keys in the `.env` file?")
