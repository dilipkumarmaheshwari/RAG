from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

PERSIST_DIR = "./chroma_db"

def load_qa_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)

    llm = Ollama(model="llama3")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa

def main():
    qa_chain = load_qa_chain()
    print("💬 Q&A App is ready (type 'exit' to quit)")

    while True:
        question = input("❓ Your question: ")
        if question.lower() in ["exit", "quit"]:
            break
        result = qa_chain(question)
        print(f"\n🧠 Answer: {result['result']}")
        print("\n📄 Source documents:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()
