from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

# --- Lazy init vars ---
rag_chain = None

def init_chain():
    """Initialize only when needed."""
    global rag_chain
    if rag_chain is None:
        from src.helper import download_embeddings
        print("Initializing model and Pinecone...")
        embeddings = download_embeddings()
        docsearch = PineconeVectorStore.from_existing_index(
            index_name="medical-chatbot",
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
        llm = ChatGroq(model="llama-3.1-8b-instant")
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("Initialization complete.")

@app.route("/health")
def health():
    return {'status': 'healthy'}, 200

@app.route("/get", methods=["POST"])
def chat():
    init_chain()
    msg = request.form.get("msg", "")
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
