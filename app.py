import streamlit as st
import nltk
import re
import fitz  # PyMuPDF
import tempfile
import os
from datetime import datetime
import pandas as pd

# Download NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config with custom theme
st.set_page_config(
    page_title="AI Resume Parser Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styles */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(145deg, #f8fafc, #e2e8f0);
        padding: 2rem;
        border-radius: 20px;
        border: 2px dashed #cbd5e0;
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        background: linear-gradient(145deg, #f1f5f9, #e2e8f0);
    }
    
    /* Card Styles */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .card-icon {
        font-size: 1.5rem;
        margin-right: 0.8rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin: 0;
    }
    
    /* Skill Tags */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease;
    }
    
    .skill-tag:hover {
        transform: scale(1.05);
    }
    
    /* Info Items */
    .info-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
    }
    
    .info-item:last-child {
        border-bottom: none;
    }
    
    .info-label {
        font-weight: 600;
        color: #4a5568;
        min-width: 80px;
        margin-right: 1rem;
    }
    
    .info-value {
        color: #2d3748;
        flex: 1;
    }
    
    /* List Items */
    .experience-item, .education-item {
        background: #f8fafc;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #667eea;
        transition: background 0.2s ease;
    }
    
    .experience-item:hover, .education-item:hover {
        background: #edf2f7;
    }
    
    /* Stats Section */
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .stat-item {
        text-align: center;
        padding: 0 1rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Raw Text Section */
    .raw-text-container {
        background: #1a202c;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.4;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Sidebar Styles */
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 500;
    }
    
    .error-message {
        background: linear-gradient(135deg, #f56565, #e53e3e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 500;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3 style="margin-top: 0;">🎯 Features</h3>
        <ul style="list-style: none; padding: 0;">
            <li>📊 Smart Information Extraction</li>
            <li>🔍 Advanced Pattern Recognition</li>
            <li>💼 Professional Parsing</li>
            <li>📱 Mobile-Friendly Interface</li>
            <li>⚡ Lightning Fast Processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Supported Formats")
    st.info("📄 PDF files up to 10MB")
    
    st.markdown("### Processing Stats")
    if 'files_processed' not in st.session_state:
        st.session_state.files_processed = 0
    st.metric("Files Processed", st.session_state.files_processed)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Resume Parser Pro</h1>
    <p>Extract key information from resumes with advanced AI-powered parsing</p>
</div>
""", unsafe_allow_html=True)

# Enhanced skills list with categories
skills_database = {
    'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin'],
    'Web Technologies': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'flask', 'django', 'spring'],
    'Data Science & AI': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'opencv'],
    'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite'],
    'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'gitlab', 'terraform'],
    'Tools & Technologies': ['data analysis', 'tableau', 'power bi', 'excel', 'jira', 'confluence', 'slack']
}

# Flatten skills list for extraction
all_skills = []
for category in skills_database.values():
    all_skills.extend(category)

def extract_name(text):
    lines = text.split('\n')[:5]  # Check first 5 lines
    name_patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$',  # Full name on separate line
        r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Name: format
        r'^([A-Z][A-Z\s]+)$'  # All caps name
    ]
    
    for line in lines:
        line = line.strip()
        for pattern in name_patterns:
            match = re.search(pattern, line)
            if match and len(match.group(1).split()) >= 2:
                return match.group(1).title()
    return None

def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None

def extract_phone(text):
    phone_patterns = [
        r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'(\+\d{1,3}[-.\s]?)?\d{10}',
        r'(\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return re.sub(r'[^\d+]', '', matches[0]) if isinstance(matches[0], str) else ''.join(matches[0])
    return None

def extract_skills(text):
    text_lower = text.lower()
    found_skills = {}
    
    for category, skills in skills_database.items():
        category_skills = []
        for skill in skills:
            if skill.lower() in text_lower:
                category_skills.append(skill)
        if category_skills:
            found_skills[category] = category_skills
    
    return found_skills

def extract_education(text):
    education_patterns = [
        r'(Bachelor(?:\'s)?|B\.?[AS]\.?|BS|BA)[\s\w]*(?:in|of)?\s*([A-Za-z\s]+)(?:from|at)?\s*([A-Za-z\s&]+(?:University|College|Institute))',
        r'(Master(?:\'s)?|M\.?[AS]\.?|MS|MA|MBA)[\s\w]*(?:in|of)?\s*([A-Za-z\s]+)(?:from|at)?\s*([A-Za-z\s&]+(?:University|College|Institute))',
        r'(Ph\.?D\.?|PhD|Doctorate)[\s\w]*(?:in|of)?\s*([A-Za-z\s]+)(?:from|at)?\s*([A-Za-z\s&]+(?:University|College|Institute))',
    ]
    
    education = []
    for pattern in education_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            education.append({
                'degree': match.group(1),
                'field': match.group(2).strip() if match.group(2) else '',
                'institution': match.group(3).strip() if match.group(3) else ''
            })
    
    return education

def extract_experience(text):
    sentences = nltk.sent_tokenize(text)
    experience = []
    exp_keywords = ['experience', 'worked', 'employed', 'position', 'role', 
                   'responsible', 'managed', 'led', 'developed', 'created',
                   'engineer', 'developer', 'analyst', 'manager', 'intern']
    
    for sent in sentences:
        if len(sent) > 20 and any(keyword.lower() in sent.lower() for keyword in exp_keywords):
            # Clean and format the sentence
            cleaned_sent = re.sub(r'\s+', ' ', sent.strip())
            if len(cleaned_sent) > 50:  # Only include substantial sentences
                experience.append(cleaned_sent)
    
    return experience[:10]  # Return top 10 most relevant

def extract_years_of_experience(text):
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience[:\s]*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:exp|experience)'
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        years.extend([int(match) for match in matches])
    
    return max(years) if years else None

# File upload section
st.markdown("""
<div class="upload-section">
    <h3>📤 Upload Your Resume</h3>
    <p>Drag and drop or click to select a PDF file</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

if uploaded_file is not None:
    try:
        # Show processing message
        with st.spinner('🔄 Processing your resume...'):
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Extract text from PDF
            text = ""
            with fitz.open(tmp_path) as doc:
                for page in doc:
                    text += page.get_text()
            
            os.unlink(tmp_path)  # Clean up temporary file
            
            # Update processing stats
            st.session_state.files_processed += 1
        
        # Show success message
        st.markdown("""
        <div class="success-message">
            ✅ Resume processed successfully!
        </div>
        """, unsafe_allow_html=True)
        
        # Extract all information
        name = extract_name(text)
        email = extract_email(text)
        phone = extract_phone(text)
        skills = extract_skills(text)
        education = extract_education(text)
        experience = extract_experience(text)
        years_exp = extract_years_of_experience(text)
        
        # Calculate stats
        total_skills = sum(len(skill_list) for skill_list in skills.values())
        
        # Display stats
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-item">
                <span class="stat-number">{total_skills}</span>
                <span class="stat-label">Skills Found</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{len(education)}</span>
                <span class="stat-label">Education Records</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{len(experience)}</span>
                <span class="stat-label">Experience Items</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{years_exp or 'N/A'}</span>
                <span class="stat-label">Years Experience</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main content in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Personal Information Card
            st.markdown(f"""
            <div class="info-card">
                <div class="card-header">
                    <span class="card-icon">👤</span>
                    <h3 class="card-title">Personal Information</h3>
                </div>
                <div class="info-item">
                    <span class="info-label">Name:</span>
                    <span class="info-value">{name or 'Not detected'}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Email:</span>
                    <span class="info-value">{email or 'Not found'}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Phone:</span>
                    <span class="info-value">{phone or 'Not found'}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Experience:</span>
                    <span class="info-value">{f'{years_exp} years' if years_exp else 'Not specified'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Education Card
            st.markdown("""
            <div class="info-card">
                <div class="card-header">
                    <span class="card-icon">🎓</span>
                    <h3 class="card-title">Education</h3>
                </div>
            """, unsafe_allow_html=True)
            
            if education:
                for edu in education:
                    st.markdown(f"""
                    <div class="education-item">
                        <strong>{edu['degree']}</strong>
                        {f" in {edu['field']}" if edu['field'] else ""}
                        {f"<br><em>{edu['institution']}</em>" if edu['institution'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<p>No formal education records detected</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Skills Card
            st.markdown("""
            <div class="info-card">
                <div class="card-header">
                    <span class="card-icon">💻</span>
                    <h3 class="card-title">Technical Skills</h3>
                </div>
            """, unsafe_allow_html=True)
            
            if skills:
                for category, skill_list in skills.items():
                    st.markdown(f"<p><strong>{category}:</strong></p>", unsafe_allow_html=True)
                    skills_html = ""
                    for skill in skill_list:
                        skills_html += f'<span class="skill-tag">{skill.title()}</span>'
                    st.markdown(skills_html, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.markdown("<p>No technical skills detected</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Experience Card
            st.markdown("""
            <div class="info-card">
                <div class="card-header">
                    <span class="card-icon">💼</span>
                    <h3 class="card-title">Work Experience</h3>
                </div>
            """, unsafe_allow_html=True)
            
            if experience:
                for i, exp in enumerate(experience[:5], 1):
                    st.markdown(f"""
                    <div class="experience-item">
                        <strong>Experience {i}:</strong><br>
                        {exp}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<p>No work experience details detected</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Raw text expander
        with st.expander("🔍 View Extracted Text", expanded=False):
            st.markdown("""
            <div class="raw-text-container">
            """, unsafe_allow_html=True)
            st.text(text[:3000] + "..." if len(text) > 3000 else text)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Download results as JSON
        if st.button("📥 Download Analysis Results", type="primary"):
            results = {
                "personal_info": {
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "years_experience": years_exp
                },
                "skills": skills,
                "education": education,
                "experience": experience,
                "analysis_date": datetime.now().isoformat(),
                "file_name": uploaded_file.name
            }
            
            st.download_button(
                label="💾 Download JSON",
                data=str(results),
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-message">
            ❌ Error processing resume: {str(e)}
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
        <h3>🚀 Ready to Parse Your Resume?</h3>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">
            Upload a PDF resume above to get started with AI-powered information extraction
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); min-width: 200px;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                <strong>Fast Processing</strong><br>
                <small>Extract info in seconds</small>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); min-width: 200px;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <strong>High Accuracy</strong><br>
                <small>Advanced pattern recognition</small>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); min-width: 200px;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <strong>Detailed Analysis</strong><br>
                <small>Comprehensive extraction</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid #e2e8f0; color: #64748b;">
    <p>Built with ❤️ using Streamlit | AI Resume Parser Pro v2.0</p>
</div>
""", unsafe_allow_html=True)