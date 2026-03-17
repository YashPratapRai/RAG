import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

st.title("📄 RAG Document Question Answering")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type="pdf",
    accept_multiple_files=True
)

query = st.text_input("Ask a question about the documents")

if uploaded_files:

    documents = []

    for file in uploaded_files:

        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyMuPDFLoader(file.name)
        documents.extend(loader.load())

    st.success(f"{len(documents)} pages loaded.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in chunks]
    embeddings = model.encode(texts)

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("rag_collection")

    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        collection.add(
            ids=[str(i)],
            documents=[text],
            embeddings=[embedding.tolist()]
        )

    if query:

        query_embedding = model.encode([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )

        context = "\n".join(results["documents"][0])

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant"
        )

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)