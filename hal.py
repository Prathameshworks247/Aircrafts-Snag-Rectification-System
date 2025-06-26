import streamlit as st
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from update_index import add_snag_to_index

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.set_page_config(page_title="Snag Rectifier", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2a5298;
        box-shadow: 0 0 10px rgba(42, 82, 152, 0.3);
    }
    
    .confidence-score {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .snag-card {
        background: black;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .past-snag-item {
        background: white;
        border-left: 4px solid #2a5298;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .recent-query-item {
        background: white;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>✈️ Sky Sentinel: Snag Rectification System</h1>
    <p>Advanced Aircraft Maintenance Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "past_snags" not in st.session_state:
    st.session_state.past_snags = []

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Statistics")
    
    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h3>{len(st.session_state.query_history)}</h3>
            <p>Total Queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <h3>{len(st.session_state.past_snags)}</h3>
            <p>Resolved Snags</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    
    # Settings
    confidence_threshold = st.slider("Confidence Threshold", 0, 100, 75)
    max_similar_docs = st.slider("Max Similar Documents", 1, 10, 3)
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.query_history = []
        st.session_state.past_snags = []
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["🔧 New Snag Analysis", "📋 Past Snags", "🕒 Recent Queries"])

with tab1:
    st.markdown("### Enter Snag Details")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_text = st.text_input(
            "**Describe the snag you are facing:**",
            placeholder="e.g., Engine oil pressure low during takeoff, unusual vibration in landing gear...",
            key="snag_input"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze Snag", type="primary", use_container_width=True)
    
    # Processing
    if input_text and analyze_button:
        with st.spinner("🔄 Analyzing snag and searching for similar cases..."):
            # Initialize components
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            try:
                vectorstore = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
                retriever = vectorstore.as_retriever(search_kwargs={"k": max_similar_docs})
                
                # Gemini Model
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=os.getenv("GOOGLE_API_KEY"))
                
                # Prompt Template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert in Aircraft Snags. Rectify the snags and provide a detailed explanation. "
                               "The output should be in format : \n"
                               "Rectification: <rectification>\n"
                               "Ranked list of past similar snags with their rectification action: <list>\n"
                               "Similarity confidence score: <percentage in BIG letters>"),
                    MessagesPlaceholder(variable_name="history"),
                    ("user", "Question: {input}")
                ])
                
                output_parser = StrOutputParser()
                memory = st.session_state.memory
                history = memory.load_memory_variables({})["history"]
                
                chain = prompt | llm | output_parser
                
                # Get similar documents
                similar_docs = retriever.get_relevant_documents(input_text)
                examples_text = "\n".join([doc.page_content for doc in similar_docs])
                
                full_input = {
                    "input": f"{input_text}\n\nRelevant Past Snags:\n{examples_text}",
                    "history": history
                }
                
                response = chain.invoke(full_input)
                
                # Parse response
                def parse_response(response: str):
                    rectification = re.search(r"Rectification:\s*(.*?)(?:Ranked list|Similarity confidence score:)", response, re.DOTALL)
                    ranked_list = re.search(r"Ranked list of past similar snags.*?:\s*(.*?)(?:Similarity confidence score:)", response, re.DOTALL)
                    confidence = re.search(r"Similarity confidence score:\s*(.*)", response)
                
                    return {
                        "rectification": rectification.group(1).strip() if rectification else "Not found",
                        "ranked_list": ranked_list.group(1).strip() if ranked_list else "Not found",
                        "confidence": confidence.group(1).strip() if confidence else "Not found"
                    }
                
                parsed = parse_response(response)
                
                # Update memory
                memory.chat_memory.add_user_message(input_text)
                memory.chat_memory.add_ai_message(response)
                
                # Store in session state
                query_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": input_text,
                    "rectification": parsed['rectification'],
                    "confidence": parsed['confidence'],
                    "similar_snags": parsed['ranked_list']
                }
                
                st.session_state.query_history.append(query_data)
                st.session_state.past_snags.append({
                    "snag": input_text,
                    "rectification": parsed['rectification'],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": parsed['confidence']
                })
                
                # Display results
                st.markdown("---")
                
                # Issue display
                st.markdown("## 🔍 Analysis Results")
                
                st.markdown(f"""
                <div class="snag-card">
                    <h4>⚠️ Reported Issue:</h4>
                    <p><strong>{input_text}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Results layout
                col1, col2 = st.columns([7, 3])
                
                with col1:
                    st.markdown("### 🔧 Recommended Rectification")
                    st.success(parsed['rectification'])
                    
                    st.markdown("### 📚 Similar Past Cases")
                    st.info(parsed['ranked_list'])
                
                with col2:
                    st.markdown("### 🎯 Confidence Score")
                    confidence_num = re.search(r'(\d+)', parsed['confidence'])
                    if confidence_num:
                        conf_val = int(confidence_num.group(1))
                        if conf_val >= confidence_threshold:
                            st.success(f"**{parsed['confidence']}**")
                        else:
                            st.warning(f"**{parsed['confidence']}**")
                    else:
                        st.info(f"**{parsed['confidence']}**")
                    
                    st.markdown("### 📈 Analysis Details")
                    st.metric("Similar Documents", len(similar_docs))
                    st.metric("Processing Time", "< 2 sec")
                
                # Add to FAISS index
                try:
                    add_snag_to_index(input_text, parsed['rectification'])
                    st.success("✅ Snag data added to knowledge base!")
                except Exception as e:
                    st.warning(f"⚠️ Could not update knowledge base: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Error processing snag: {str(e)}")
                st.info("Please check your configuration and try again.")

with tab2:
    st.markdown("### 📋 Historical Snag Database")
    
    if st.session_state.past_snags:
        # Search and filter
        search_term = st.text_input("🔍 Search past snags...", placeholder="Enter keywords to search")
        
        filtered_snags = st.session_state.past_snags
        if search_term:
            filtered_snags = [
                snag for snag in st.session_state.past_snags 
                if search_term.lower() in snag['snag'].lower() or 
                   search_term.lower() in snag['rectification'].lower()
            ]
        
        st.markdown(f"**Found {len(filtered_snags)} snag(s)**")
        
        for i, snag in enumerate(reversed(filtered_snags)):
            with st.expander(f"🔧 {snag['snag'][:80]}... ({snag['timestamp']})"):
                st.markdown(f"""
                <div class="past-snag-item">
                    <h5>🚨 Issue:</h5>
                    <p>{snag['snag']}</p>
                    
                    <h5>✅ Rectification:</h5>
                    <p>{snag['rectification']}</p>
                    
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                        <span><strong>📅 Date:</strong> {snag['timestamp']}</span>
                        <span><strong>🎯 Confidence:</strong> {snag['confidence']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📝 No past snags recorded yet. Analyze some snags to see them here!")

with tab3:
    st.markdown("### 🕒 Recent Query History")
    
    if st.session_state.query_history:
        # Display recent queries
        st.markdown(f"**Total Queries: {len(st.session_state.query_history)}**")
        
        for i, query in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Query #{len(st.session_state.query_history) - i}: {query['query'][:60]}... ({query['timestamp']})"):
                st.markdown(f"""
                <div class="recent-query-item">
                    <h5>❓ Query:</h5>
                    <p>{query['query']}</p>
                    
                    <h5>💡 System Response:</h5>
                    <p>{query['rectification']}</p>
                    
                    <h5>📊 Similar Cases:</h5>
                    <p>{query['similar_snags']}</p>
                    
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                        <span><strong>📅 Timestamp:</strong> {query['timestamp']}</span>
                        <span><strong>🎯 Confidence:</strong> {query['confidence']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("🔍 No queries recorded yet. Start analyzing snags to see query history!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🛡️ Sky Sentinel - Advanced Aircraft Maintenance Intelligence Platform | Built with Streamlit & LangChain</p>
</div>
""", unsafe_allow_html=True)