import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Load and validate your CSV
df = pd.read_csv("snag_data.csv")
print(df.columns)

df['combined'] = "Snag: " + df['Snag'] + " | Fix: " + df['Rectification']
documents = [Document(page_content=row) for row in df['combined']]

# Initialize embedder
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and save vectorstore
vectorstore = FAISS.from_documents(documents, embedder)
vectorstore.save_local("faiss_index")

print("✅ Vectorstore saved at ./faiss_index/")

import os

def add_snag_to_index(snag, rectification):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_doc = Document(page_content=snag + "\nRectification: " + rectification)

    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        vectorstore = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents([new_doc], embedder)
    
    vectorstore.add_documents([new_doc])
    vectorstore.save_local(index_dir)
