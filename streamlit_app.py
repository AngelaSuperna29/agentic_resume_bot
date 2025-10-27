import streamlit as st
import os
from src import ingestion, matcher

st.set_page_config(page_title="Agentic Resume Matcher", page_icon="🧠", layout="wide")

st.title("🧠 Agentic Resume Matcher")
st.write("Upload resumes and match them to job descriptions using AI.")

# Ensure folders exist
os.makedirs("data/resumes", exist_ok=True)
os.makedirs("data/resumes_text", exist_ok=True)

# Sidebar
st.sidebar.header("📄 Resume Management")
uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True, type=["pdf"])

if st.sidebar.button("Ingest Uploaded Resumes"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data/resumes", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        ingestion.ingest_all_from_folder("data/resumes")
        st.sidebar.success("✅ Resumes ingested successfully!")
    else:
        st.sidebar.warning("⚠️ Please upload at least one PDF file.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Rebuild Index"):
    from src import indexer
    indexer.build_index()
    st.sidebar.success("✅ Index rebuilt successfully!")

# Job matching section
st.header("💼 Match Job Description")
job_desc = st.text_area("Paste the job description here:", height=200)

if st.button("🔍 Match Candidates"):
    if not job_desc.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        results = matcher.match_job(job_desc, top_k=5)
        if results:
            st.success("✅ Matching complete. Here are the top candidates:")
            for i, (resume_name, score, summary) in enumerate(results, start=1):
                st.markdown(f"### 🧾 {i}. {resume_name}")
                st.markdown(f"**Match Score:** `{score:.2f}`")
                st.markdown(f"**Summary:** {summary}")
                st.markdown("---")
        else:
            st.info("No matches found. Try rebuilding the index or ingesting resumes.")
