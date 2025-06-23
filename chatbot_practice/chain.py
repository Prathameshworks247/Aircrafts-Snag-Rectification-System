from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template with memory placeholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in Aircraft Snags. Rectify the snags and provide a detailed explanation. The output should be in format : Rectification: <rectification> Ranked list of past similar snags with their rectification action: <Ranked list of past similar snags with their rectification action> Similarity confidence score: <just in percentage in big letter>"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "Question: {input}")
    ]
)

# Initialize memory once
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Streamlit UI
st.set_page_config(page_title="Snag Rectifier", page_icon="🛠️", layout="wide")
st.markdown("<h1 style=' text-align: center;display: flex; justify-content: center;'>Snag Rectification Recommender System</h1>", unsafe_allow_html=True)
# Custom CSS to center and resize the input box
st.markdown("""
    <style>
    div[data-testid="stTextInput"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    div[data-testid="stTextInput"] > div {
        width: 800px;  /* Adjust as needed */
    }
    </style>
""", unsafe_allow_html=True)

# Regular Streamlit input
input_text = st.text_input("**What is the issue? Describe the snag you are facing.**")


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

def parse_response(response: str):
    rectification = re.search(r"Rectification:\s*(.*?)(?:Ranked list|Similarity confidence score:)", response, re.DOTALL)
    ranked_list = re.search(r"Ranked list of past similar snags.*?:\s*(.*?)(?:Similarity confidence score:)", response, re.DOTALL)
    confidence = re.search(r"Similarity confidence score:\s*(.*)", response)

    return {
        "rectification": rectification.group(1).strip() if rectification else "Not found",
        "ranked_list": ranked_list.group(1).strip() if ranked_list else "Not found",
        "confidence": confidence.group(1).strip() if confidence else "Not found"
    }

if input_text:
    memory = st.session_state.memory
    history = memory.load_memory_variables({})["history"]
    full_input = {"input": input_text, "history": history}

    # Generate response
    response = chain.invoke(full_input)

    # Update memory
    memory.chat_memory.add_user_message(input_text)
    memory.chat_memory.add_ai_message(response)

    # Parse response
    parsed = parse_response(response)

    # Display current interaction only
    st.markdown("## ⛓️‍💥 Issue:")
    st.markdown(f"> ### {input_text}")
    col1, col2 = st.columns([8,3])
    with col2:
        st.markdown("#### 🔍 Confidence Score")
        st.success(f"{parsed['confidence']}")
    with col1:
        st.markdown("## 👨‍🔧 System Suggestion:")
        st.markdown(f"#### Rectification:") 
        st.info(f"{parsed['rectification']}")
        st.markdown(f"#### Similar Past Snags:")
        st.info(f"{parsed['ranked_list']}")
