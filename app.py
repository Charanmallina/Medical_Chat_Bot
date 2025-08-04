from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_embeddings
from src.prompt import system_prompt

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# --- Lazy-loaded global variables ---
embeddings = None
docsearch = None
rag_chain = None


def init_chain():
    """Initialize embeddings, Pinecone, and the RAG chain (lazy-loaded)."""
    global embeddings, docsearch, rag_chain
    if rag_chain is None:
        print("Initializing embeddings and RAG chain...")
        embeddings = download_embeddings()

        index_name = "medical-chatbot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        llm = ChatGroq(model="llama-3.1-8b-instant")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("RAG chain initialized successfully.")


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/health")
def health_check():
    return {'status': 'healthy'}, 200


@app.route("/get", methods=["GET", "POST"])
def chat():
    init_chain()  # Ensure the chain is ready
    msg = request.form["msg"]
    print(f"Received message: {msg}")
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
