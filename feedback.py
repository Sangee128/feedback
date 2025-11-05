import streamlit as st
import pandas as pd
import io
from langchain_google_genai import ChatGoogleGenerativeAI
import json



# ============================================================================
# CONFIGURATION
# ============================================================================
GEMINI_API_KEY = "AIzaSyBMEqNFTqBIgAMci-R8UkT05-xmHHEV0Bc"



st.set_page_config(
    page_title="Feedback Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: #f8f9ff;
        margin: 1rem 0;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )



llm = get_llm()



# ============================================================================
# FILE READING FUNCTIONS
# ============================================================================
def read_csv(file_content: bytes) -> pd.DataFrame:
    try:
        # Try UTF-8 first
        return pd.read_csv(io.BytesIO(file_content))
    except UnicodeDecodeError:
        # Retry with fallback encodings
        try:
            return pd.read_csv(io.BytesIO(file_content), encoding="latin1")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None




def read_excel(file_content: bytes) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        return df
    except Exception as e:
        st.error(f"Error reading Excel: {str(e)}")
        return None



def read_txt(file_content: bytes) -> pd.DataFrame:
    try:
        content = file_content.decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        df = pd.DataFrame({'feedback': lines})
        return df
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return None



def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    content = uploaded_file.read()
    if filename.endswith('.csv'):
        return read_csv(content)
    elif filename.endswith(('.xlsx', '.xls')):
        return read_excel(content)
    elif filename.endswith('.txt'):
        return read_txt(content)
    else:
        st.error("❌ Unsupported format. Please upload CSV, Excel, or TXT file")
        return None



# ============================================================================
# FILE UNDERSTANDING FUNCTIONS
# ============================================================================
def understand_file_structure(df: pd.DataFrame) -> dict:
    with st.spinner("🔍 Understanding file structure..."):
        sample_data = df.head(10).to_string()
        column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
        prompt = f"""Analyze this dataset and identify the feedback/comment columns.



**Dataset Info:**
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}



**Column Names and Types:**
{column_info}



**Sample Data (first 10 rows):**
{sample_data}



**Your Task:**
1. Identify which columns contain feedback/comments/reviews/responses
2. Determine the feedback type (customer feedback, student feedback, employee feedback, survey responses, etc.)
3. Identify any metadata columns (name, email, date, rating, etc.)
4. Suggest the best approach to extract and analyze the feedback
5. Suggest improvements if needed



**Response Format (JSON):**
{{
    "feedback_type": "customer/student/employee/survey/other",
    "feedback_columns": ["col1", "col2"],
    "metadata_columns": ["name", "date", "rating"],
    "description": "Brief description of what this feedback is about",
    "suggested_analysis": "What aspects to focus on in analysis"
}}



Respond ONLY with valid JSON, no other text."""



        response = llm.invoke(prompt)
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            structure = json.loads(content)
            return structure
        except Exception as e:
            st.warning(f"Could not parse LLM response. Using fallback analysis. Error: {str(e)}")
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            return {
                "feedback_type": "general",
                "feedback_columns": text_columns,
                "metadata_columns": [],
                "description": "General feedback data",
                "suggested_analysis": "Standard sentiment and theme analysis"
            }



def extract_feedbacks(df: pd.DataFrame, feedback_columns: list):
    """Extract all feedbacks from identified columns"""
    feedbacks = []
    for col in feedback_columns:
        if col in df.columns:
            col_feedbacks = df[col].dropna().astype(str).tolist()
            col_feedbacks = [f.strip() for f in col_feedbacks if f.strip() and f.lower() != 'nan']
            feedbacks.extend(col_feedbacks)
    
    return feedbacks



# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_feedbacks(feedbacks: list, feedback_type: str, suggested_analysis: str, description: str) -> dict:
    """Analyze feedbacks with context-aware prompting"""
    if not feedbacks:
        return {"error": "No feedback found"}
    
    feedback_text = "\n".join([f"{i+1}. {fb}" for i, fb in enumerate(feedbacks)])
    
    if feedback_type.lower() == "student":
        categories = """
   - Teaching quality feedback
   - Course content feedback
   - Facility/infrastructure feedback
   - Positive comments
   - Suggestions for improvement
   - Complaints"""
    elif feedback_type.lower() == "employee":
        categories = """
   - Work environment feedback
   - Management feedback
   - Benefits and compensation
   - Career development
   - Positive aspects
   - Areas of concern"""
    elif feedback_type.lower() == "customer":
        categories = """
   - Product/service quality
   - Customer service experience
   - Pricing concerns
   - Feature requests
   - Bug reports
   - Compliments"""
    elif feedback_type.lower() == "product":
        categories = """
   - product features
   - usability
   - design
   - performance
   - bugs/issues
   - feature requests"""
    else:
        categories = """
   - Positive feedback
   - Negative feedback
   - Suggestions
   - Complaints
   - Questions
   - Feature requests"""
    
    prompt = f"""You are analyzing **{feedback_type.upper()} FEEDBACK**.



**Context:** {description}



**Focus Areas:** {suggested_analysis}



**Analysis Requirements:**



1. **Sentiment Breakdown**
   - Positive count and percentage
   - Negative count and percentage  
   - Neutral count and percentage
   - Overall sentiment trend



2. **Main Themes** (top 5-7 recurring topics with examples)



3. **Category Breakdown**
{categories}



4. **Critical Issues** (most urgent problems to address)



5. **Positive Highlights** (what's working well)



6. **Top 5 Action Items** (prioritized recommendations)



7. **Key Insights** (surprising or important patterns)



**Feedbacks to analyze ({len(feedbacks)} total):**



{feedback_text}



---



**Format:** Use clear markdown with headers, bullet points, and **bold** for emphasis. Be specific and actionable."""
    
    with st.spinner("🤖 AI is analyzing your feedback..."):
        response = llm.invoke(prompt)
    
    return {
        "analyzed_count": len(feedbacks),
        "analysis": response.content,
        "feedback_type": feedback_type
    }



# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    st.title("Intelligent Feedback Analyzer")
    st.markdown("### Upload any feedback file → AI understands and analyzes it automatically")
    
    with st.sidebar:
        st.header("About")
        st.info("""
        **Smart Features:**
        - Auto-detects feedback type
        - Understands file structure
        - Context-aware analysis
        - Comprehensive insights
        
        **Supported Formats:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - Text (.txt)
        """)
        
        st.header("🔧 Settings")
        max_feedbacks = st.slider(
            "Max feedbacks to analyze",
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )
        
        st.markdown("---")
        st.caption("Powered by Google Gemini 2.0 Flash")
    
    uploaded_file = st.file_uploader(
        "Upload your feedback file",
        type=['csv', 'xlsx', 'xls', 'txt'],
        help="Upload any file containing feedback data"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        df = read_uploaded_file(uploaded_file)
        
        if df is not None:
            st.markdown("---")
            
            with st.expander("Preview Raw Data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
                st.caption(f"Showing first 20 rows of {len(df)} total rows")
            
            st.markdown("### 🔍 Step 1: Understanding Your Data")
            structure = understand_file_structure(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Detected Information")
                st.markdown(f"""
                - **Feedback Type:** `{structure['feedback_type'].title()}`
                - **Description:** {structure['description']}
                - **Feedback Columns:** {', '.join([f'`{col}`' for col in structure['feedback_columns']])}
                """)
            
            with col2:
                st.markdown("#### Analysis Strategy")
                st.info(structure['suggested_analysis'])
            
            if structure.get('metadata_columns'):
                st.markdown(f"**Metadata Columns:** {', '.join([f'`{col}`' for col in structure['metadata_columns']])}")
            
            st.markdown("---")
            
            st.markdown("### Step 2: Extracting Feedback")
            feedbacks = extract_feedbacks(df, structure['feedback_columns'])
            
            if not feedbacks:
                st.error("No valid feedback found in the identified columns")
                return
            
            # LOGIC: Total feedbacks = file rows (excluding header)
            # Will analyze = min(total feedbacks, slider value)
            total_feedbacks = len(df)
            will_analyze = min(total_feedbacks, max_feedbacks)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Feedbacks", total_feedbacks)
                
            with col2:
                st.metric("Will Analyze", will_analyze)
            with col3:
                st.metric("Feedback Columns", len(structure['feedback_columns']))
            
            with st.expander("Sample Feedbacks", expanded=False):
                for i, fb in enumerate(feedbacks[:10], 1):
                    st.markdown(f"**{i}.** {fb}")
            
            st.markdown("---")
            
            st.markdown("### Step 3: AI Analysis")
            
            if st.button("Start AI Analysis", type="primary", use_container_width=True):
                limited_feedbacks = feedbacks[:will_analyze]
                results = analyze_feedbacks(
                    limited_feedbacks,
                    structure['feedback_type'],
                    structure['suggested_analysis'],
                    structure['description']
                )
                
                if "error" in results:
                    st.error(results["error"])
                    return
                
                # Add total feedbacks (file rows) to results
                results['total_feedbacks'] = total_feedbacks
                
                st.session_state['analysis_results'] = results
                st.session_state['structure'] = structure
            
            if 'analysis_results' in st.session_state:
                results = st.session_state['analysis_results']
                structure = st.session_state['structure']
                
                st.markdown("---")
                st.markdown("## Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Feedbacks", results['total_feedbacks'])
                with col2:
                    st.metric("Analyzed", results['analyzed_count'])
                with col3:
                    coverage = (results['analyzed_count'] / results['total_feedbacks'] * 100)
                    st.metric("Coverage", f"{coverage:.1f}%")
                
                st.markdown("---")
                
                st.markdown("### Detailed Analysis Report")
                st.markdown(results['analysis'])
                
                st.markdown("---")
                st.markdown("### Export Results")
                
                report = f"""# Feedback Analysis Report



**File:** {uploaded_file.name}
**Feedback Type:** {structure['feedback_type'].title()}
**Total Feedbacks:** {results['total_feedbacks']}
**Analyzed:** {results['analyzed_count']}
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}



---



## File Understanding



**Description:** {structure['description']}



**Feedback Columns:** {', '.join(structure['feedback_columns'])}



**Analysis Strategy:** {structure['suggested_analysis']}



---



## AI Analysis



{results['analysis']}



---



*Generated by Intelligent Feedback Analyzer*
"""
                
                st.download_button(
                    label="📥 Download Full Report (Markdown)",
                    data=report,
                    file_name=f"feedback_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )



if __name__ == "__main__":

    main()
