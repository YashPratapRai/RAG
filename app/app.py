import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

st.title("📄 Conversational RAG Document QA")

# Conversation Buffer Window Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    client = chromadb.Client(
        Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(
        "rag_collection"
    )

    # Avoid duplicate IDs on reruns
    try:
        existing_count = collection.count()
    except:
        existing_count = 0

    if existing_count == 0:

        for i, (text, embedding) in enumerate(
            zip(texts, embeddings)
        ):
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

        context = "\n".join(
            results["documents"][0]
        )

        # Last 5 conversations (Window Memory)
        history = ""

        for chat in st.session_state.chat_history[-5:]:

            history += f"""
User: {chat['question']}
Assistant: {chat['answer']}
"""

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0
        )

        messages = [

            SystemMessage(
                content="""
You are a helpful document question answering assistant.

Use the provided document context to answer.

If the answer is not present in the document,
say:
'I could not find the answer in the uploaded documents.'
"""
            ),

            HumanMessage(
                content=f"""
Previous Conversation:
{history}

Document Context:
{context}

Current Question:
{query}
"""
            )
        ]

        response = llm.invoke(messages)

        # Save conversation
        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": response.content
            }
        )

        st.subheader("Answer")
        st.write(response.content)

        # Show recent conversation
        if st.session_state.chat_history:

            st.subheader("Recent Conversation")

            for i, chat in enumerate(
                st.session_state.chat_history[-5:],
                start=1
            ):
                st.write(
                    f"**Q{i}:** {chat['question']}"
                )
                st.write(
                    f"**A{i}:** {chat['answer']}"
                )
                st.write("---")