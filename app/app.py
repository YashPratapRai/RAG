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

st.set_page_config(
    page_title="Conversational RAG QA",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Conversational RAG Document QA")

# ----------------------------
# Session State
# ----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# PDF Upload
# ----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type="pdf",
    accept_multiple_files=True
)

# ----------------------------
# Clear Chat Button
# ----------------------------

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# ----------------------------
# Display Previous Chats
# ----------------------------

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

# ----------------------------
# Process PDFs
# ----------------------------

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

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        doc.page_content
        for doc in chunks
    ]

    embeddings = model.encode(texts)

    client = chromadb.Client(
        Settings(
            anonymized_telemetry=False
        )
    )

    collection = client.get_or_create_collection(
        "rag_collection"
    )

    # Prevent duplicate insertion
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
                embeddings=[
                    embedding.tolist()
                ]
            )

    # ----------------------------
    # Chat Input
    # ----------------------------

    query = st.chat_input(
        "Ask a question about your documents..."
    )

    if query:

        # Show User Message

        with st.chat_message("user"):
            st.write(query)

        # Query Embedding

        query_embedding = model.encode(
            [query]
        )[0]

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=3
        )

        context = "\n".join(
            results["documents"][0]
        )

        # ----------------------------
        # Buffer Window Memory
        # Last 5 Conversations
        # ----------------------------

        history = ""

        for chat in st.session_state.chat_history[-5:]:

            history += f"""
User: {chat['question']}
Assistant: {chat['answer']}
"""

        # ----------------------------
        # Chat Model
        # ----------------------------

        llm = ChatGroq(
            api_key=os.getenv(
                "GROQ_API_KEY"
            ),
            model="llama-3.1-8b-instant",
            temperature=0
        )

        messages = [

            SystemMessage(
                content="""
You are a helpful document
question answering assistant.

Use ONLY the provided
document context.

Use previous conversation
history when answering
follow-up questions.

If the answer is not found
in the documents, reply:

'I could not find the answer
in the uploaded documents.'
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

        response = llm.invoke(
            messages
        )

        # Show Assistant Message

        with st.chat_message(
            "assistant"
        ):
            st.write(
                response.content
            )

        # Save Conversation

        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": response.content
            }
        )

# ----------------------------
# Expandable Chat History
# ----------------------------

if st.session_state.chat_history:

    with st.expander(
        "📜 View Chat History"
    ):

        for i, chat in enumerate(
            st.session_state.chat_history,
            start=1
        ):

            st.markdown(
                f"**Q{i}:** {chat['question']}"
            )

            st.markdown(
                f"**A{i}:** {chat['answer']}"
            )

            st.markdown("---")