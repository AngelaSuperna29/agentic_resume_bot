import streamlit as st
from pipeline import ResumeRAG
from resume_parser import parse_resume

st.set_page_config(page_title='Agentic Resume Bot (Local)', layout='wide')
st.title('Agentic Resume Bot — Local RAG Screening (Ollama + FAISS)')

st.sidebar.header('Upload Resumes')
uploaded_files = st.sidebar.file_uploader('Upload resumes (.txt, .pdf, .docx)', accept_multiple_files=True)
job_file = st.sidebar.file_uploader('Upload a job description (.txt)', type=['txt'])

if 'model' not in st.session_state:
    st.session_state['model'] = None

if st.sidebar.button('Build / Rebuild Index'):
    if not uploaded_files:
        st.sidebar.error('Please upload at least one resume file')
    else:
        resume_texts = []
        for f in uploaded_files:
            with open(f.name, 'wb') as out:
                out.write(f.getbuffer())
            txt = parse_resume(f.name)
            meta = {'filename': f.name}
            resume_texts.append((txt, meta))

        with st.spinner('Building embeddings and index...'):
            rag = ResumeRAG()
            rag.build(resume_texts)
            st.session_state['model'] = rag
        st.success('Index built successfully ✅')

if st.session_state['model']:
    rag: ResumeRAG = st.session_state['model']
    st.header('Job Description & Ranking')
    jd_text = ''
    if job_file:
        jd_text = job_file.getvalue().decode('utf-8')
        st.subheader('Uploaded Job Description')
        st.write(jd_text)
    else:
        jd_text = st.text_area('Paste job description here', height=200)

    if st.button('Rank Candidates'):
        if not jd_text.strip():
            st.error('Please provide a job description.')
        else:
            with st.spinner('Ranking candidates...'):
                ranked = rag.rank_candidates(jd_text, top_k=10)
            st.success('Ranking complete:')
            for i, r in enumerate(ranked, start=1):
                st.markdown(f"**{i}. {r['metadata'].get('filename','Candidate')} — score {r['score']:.4f}**")
                st.write(r['text_snippet'])
else:
    st.info('Upload resumes and click "Build / Rebuild Index" to start.')
