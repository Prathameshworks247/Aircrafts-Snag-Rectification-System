import streamlit as st
import os
import re
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

st.set_page_config(page_title="Snag Rectifier", layout="wide")
st.markdown("<h1 style='text-align: center;'>Snag Rectification Recommender System</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stTextInput"] {
    display: flex;
    flex-direction: column;
    align-items: center;
}
div[data-testid="stTextInput"] > div {
    width: 800px;
}
</style>
""", unsafe_allow_html=True)

input_text = st.text_input("**What is the issue? Describe the snag you are facing.**")

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Embedding and vector store
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
    similar_docs = retriever.get_relevant_documents(input_text)
    examples_text = "\n".join([doc.page_content for doc in similar_docs])
    
    full_input = {
        "input": f"{input_text}\n\nRelevant Past Snags:\n{examples_text}",
        "history": history
    }

    response = chain.invoke(full_input)

    memory.chat_memory.add_user_message(input_text)
    memory.chat_memory.add_ai_message(response)

    parsed = parse_response(response)

    # Display
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

    # Append new data to FAISS
    add_snag_to_index(input_text, parsed['rectification'])
