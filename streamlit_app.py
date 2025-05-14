import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

PERSIST_DIR = "./chroma_db"

@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    llm = Ollama(model="llama3")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa

# UI
st.set_page_config(page_title="🧠 Local Q&A App", layout="centered")
st.title("🧠 Local Q&A Chatbot")
st.markdown("Ask questions based on documents in your ChromaDB.")

query = st.text_input("💬 Enter your question:")

if query:
    with st.spinner("🤖 Thinking..."):
        qa_chain = load_qa_chain()
        result = qa_chain(query)

        st.success("✅ Answer:")
        st.write(result["result"])

        with st.expander("📄 Source Documents"):
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown source")
                st.markdown(f"- **{source}**\n\n{doc.page_content[:300]}...")

