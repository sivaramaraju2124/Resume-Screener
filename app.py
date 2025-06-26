import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
import re
from collections import Counter
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-card {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .reject-card {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.4

# Function to extract text from PDF resume
@st.cache_data
def extract_resume_text(file):
    try:
        return extract_text(file)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Enhanced similarity calculation with keyword matching
def calculate_enhanced_similarity(resume_text, job_desc):
    # TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    tfidf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    # Keyword matching
    job_keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', job_desc.lower()))
    resume_keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', resume_text.lower()))
    
    common_keywords = job_keywords.intersection(resume_keywords)
    keyword_score = len(common_keywords) / len(job_keywords) if job_keywords else 0
    
    # Combined score
    final_score = (tfidf_score * 0.7) + (keyword_score * 0.3)
    
    return {
        'final_score': final_score,
        'tfidf_score': tfidf_score,
        'keyword_score': keyword_score,
        'common_keywords': list(common_keywords),
        'total_keywords': len(job_keywords),
        'matched_keywords': len(common_keywords)
    }

# Function to extract key skills and experience
def extract_key_info(text):
    # Common skills patterns
    skills_patterns = [
        r'python|java|javascript|react|angular|vue|node\.js|express',
        r'sql|mysql|postgresql|mongodb|redis|elasticsearch',
        r'aws|azure|gcp|docker|kubernetes|jenkins|git',
        r'machine learning|deep learning|ai|data science|analytics',
        r'project management|agile|scrum|leadership|communication'
    ]
    
    found_skills = []
    for pattern in skills_patterns:
        matches = re.findall(pattern, text.lower())
        found_skills.extend(matches)
    
    # Experience extraction (simplified)
    experience_pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)'
    experience_matches = re.findall(experience_pattern, text.lower())
    
    return {
        'skills': list(set(found_skills)),
        'experience_years': experience_matches
    }

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Resume Screener Pro</h1>
    <p>Intelligent resume screening with detailed analysis and insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Threshold slider
    st.session_state.threshold = st.slider(
        "Matching Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=st.session_state.threshold,
        step=0.05,
        help="Adjust the minimum score required for selection"
    )
    
    # Analysis mode
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Standard", "Detailed", "Quick"],
        help="Choose analysis depth"
    )
    
    # Show statistics
    st.header("📊 Session Stats")
    if st.session_state.analysis_history:
        total_resumes = len(st.session_state.analysis_history)
        selected_resumes = sum(1 for r in st.session_state.analysis_history if r['selected'])
        
        st.metric("Total Analyzed", total_resumes)
        st.metric("Selected", selected_resumes)
        st.metric("Selection Rate", f"{(selected_resumes/total_resumes)*100:.1f}%")
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.analysis_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Resume Upload")
    uploaded_resume = st.file_uploader(
        "Upload Resume (PDF)", 
        type=["pdf"],
        help="Upload a PDF resume for analysis"
    )
    
    if uploaded_resume:
        st.success(f"✅ Uploaded: {uploaded_resume.name}")
        
        # Show file details
        file_details = {
            "Filename": uploaded_resume.name,
            "File size": f"{uploaded_resume.size / 1024:.1f} KB",
            "File type": uploaded_resume.type
        }
        
        with st.expander("📋 File Details"):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")

with col2:
    st.header("📝 Job Description")
    job_description = st.text_area(
        "Paste Job Description", 
        height=200,
        placeholder="Paste the job description here...",
        help="Enter the complete job description for comparison"
    )
    
    if job_description:
        word_count = len(job_description.split())
        st.info(f"📊 Word count: {word_count}")

# Real-time analysis trigger
if uploaded_resume and job_description:
    # Auto-analyze when both inputs are provided
    if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Extract text
        status_text.text("📄 Extracting text from PDF...")
        progress_bar.progress(25)
        resume_text = extract_resume_text(uploaded_resume)
        
        if resume_text:
            # Step 2: Analyze similarity
            status_text.text("🔍 Analyzing similarity...")
            progress_bar.progress(50)
            
            if analysis_mode == "Quick":
                time.sleep(0.5)
            elif analysis_mode == "Detailed":
                time.sleep(2)
            else:
                time.sleep(1)
            
            analysis_results = calculate_enhanced_similarity(resume_text, job_description)
            
            # Step 3: Extract key information
            status_text.text("📊 Extracting key information...")
            progress_bar.progress(75)
            key_info = extract_key_info(resume_text)
            
            # Step 4: Generate insights
            status_text.text("💡 Generating insights...")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            score = analysis_results['final_score']
            is_selected = score >= st.session_state.threshold
            
            # Main result card
            if is_selected:
                st.markdown(f"""
                <div class="success-card">
                    <h2>✅ SELECTED</h2>
                    <h3>Match Score: {score:.2%}</h3>
                    <p>This resume meets the requirements!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="reject-card">
                    <h2>❌ NOT SELECTED</h2>
                    <h3>Match Score: {score:.2%}</h3>
                    <p>This resume doesn't meet the minimum threshold of {st.session_state.threshold:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Overall Score", 
                    f"{score:.1%}",
                    delta=f"{(score - st.session_state.threshold):.1%}" if score > st.session_state.threshold else f"{(score - st.session_state.threshold):.1%}"
                )
            
            with col2:
                st.metric(
                    "TF-IDF Score", 
                    f"{analysis_results['tfidf_score']:.1%}"
                )
            
            with col3:
                st.metric(
                    "Keyword Match", 
                    f"{analysis_results['keyword_score']:.1%}"
                )
            
            with col4:
                st.metric(
                    "Keywords Found", 
                    f"{analysis_results['matched_keywords']}/{analysis_results['total_keywords']}"
                )
            
            # Detailed analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Score Breakdown", "🔍 Keywords", "💼 Skills", "📈 Visualization"])
            
            with tab1:
                st.subheader("Score Components")
                
                # Score breakdown chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=['TF-IDF Similarity', 'Keyword Matching', 'Final Score'],
                        y=[analysis_results['tfidf_score'], analysis_results['keyword_score'], score],
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                ])
                fig.update_layout(
                    title="Score Breakdown",
                    yaxis_title="Score",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Keyword Analysis")
                
                if analysis_results['common_keywords']:
                    # Most common keywords
                    keyword_freq = Counter(analysis_results['common_keywords'])
                    top_keywords = keyword_freq.most_common(10)
                    
                    if top_keywords:
                        keywords_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
                        
                        fig = px.bar(
                            keywords_df, 
                            x='Keyword', 
                            y='Frequency',
                            title="Top Matching Keywords"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Keywords list
                    st.write("**Matched Keywords:**")
                    keyword_cols = st.columns(3)
                    for i, keyword in enumerate(analysis_results['common_keywords'][:15]):
                        with keyword_cols[i % 3]:
                            st.write(f"✓ {keyword}")
                else:
                    st.warning("No common keywords found between resume and job description.")
            
            with tab3:
                st.subheader("Skills & Experience")
                
                if key_info['skills']:
                    st.write("**Technical Skills Found:**")
                    skills_cols = st.columns(2)
                    for i, skill in enumerate(key_info['skills']):
                        with skills_cols[i % 2]:
                            st.write(f"🔧 {skill}")
                
                if key_info['experience_years']:
                    st.write("**Experience Mentioned:**")
                    for exp in key_info['experience_years']:
                        st.write(f"📅 {exp} years")
                
                if not key_info['skills'] and not key_info['experience_years']:
                    st.info("No specific skills or experience patterns detected.")
            
            with tab4:
                st.subheader("Score Visualization")
                
                # Gauge chart for overall score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Match Score (%)"},
                    delta = {'reference': st.session_state.threshold * 100},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ffcccc"},
                            {'range': [40, 70], 'color': "#ffffcc"},
                            {'range': [70, 100], 'color': "#ccffcc"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': st.session_state.threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Save to history
            st.session_state.analysis_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': uploaded_resume.name,
                'score': score,
                'selected': is_selected,
                'threshold': st.session_state.threshold
            })
            
            # Download results
            if st.button("📥 Download Analysis Report"):
                report_data = {
                    'Resume': uploaded_resume.name,
                    'Analysis Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Overall Score': f"{score:.2%}",
                    'TF-IDF Score': f"{analysis_results['tfidf_score']:.2%}",
                    'Keyword Score': f"{analysis_results['keyword_score']:.2%}",
                    'Result': 'SELECTED' if is_selected else 'NOT SELECTED',
                    'Threshold Used': f"{st.session_state.threshold:.1%}",
                    'Keywords Matched': f"{analysis_results['matched_keywords']}/{analysis_results['total_keywords']}",
                    'Common Keywords': ', '.join(analysis_results['common_keywords'][:10])
                }
                
                report_df = pd.DataFrame([report_data])
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Analysis history
if st.session_state.analysis_history:
    st.markdown("---")
    st.header("📚 Analysis History")
    
    history_df = pd.DataFrame(st.session_state.analysis_history)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_score = history_df['score'].mean()
        st.metric("Average Score", f"{avg_score:.1%}")
    
    with col2:
        selection_rate = (history_df['selected'].sum() / len(history_df)) * 100
        st.metric("Selection Rate", f"{selection_rate:.1f}%")
    
    with col3:
        st.metric("Total Analyzed", len(history_df))
    
    # History table
    display_df = history_df.copy()
    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.1%}")
    display_df['selected'] = display_df['selected'].apply(lambda x: "✅ Selected" if x else "❌ Rejected")
    display_df['threshold'] = display_df['threshold'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        display_df[['timestamp', 'filename', 'score', 'selected', 'threshold']],
        use_container_width=True,
        column_config={
            'timestamp': 'Analysis Time',
            'filename': 'Resume File',
            'score': 'Score',
            'selected': 'Result',
            'threshold': 'Threshold'
        }
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 AI Resume Screener Pro | Built with Streamlit</p>
    <p>Upload resumes and get instant, detailed analysis with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)