from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from threading import Thread
from pinecone import Pinecone  # Updated Pinecone import

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

app = Flask(__name__)
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

# Lazy init variables
rag_chain = None

def init_chain():
    global rag_chain
    if rag_chain is None:
        try:
            from src.helper import download_embeddings
            print(">>> Starting model download...")
            embeddings = download_embeddings()
            print(">>> Model loaded.")

            print(">>> Connecting to Pinecone...")
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            if "medical-chatbot" not in pc.list_indexes().names():
                raise Exception("Pinecone index 'medical-chatbot' not found.")
            docsearch = PineconeVectorStore.from_existing_index(
                index_name="medical-chatbot",
                embedding=embeddings
            )
            print(">>> Pinecone connected.")

            retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            print(">>> Initializing Groq model...")
            llm = ChatGroq(model="llama-3.1-8b-instant")
            print(">>> Groq model initialized.")

            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            print(">>> Initialization complete.")
        except Exception as e:
            print(f"!!! ERROR during init_chain: {e}")
            raise

# --- Routes ---
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/health")
def health():
    return jsonify({'status': 'healthy'}), 200

# Asynchronous warmup to avoid timeout
@app.route("/warmup", methods=["POST", "GET"])
def warmup():
    def background_init():
        try:
            init_chain()
        except Exception as e:
            print(f"Warmup error: {e}")
    Thread(target=background_init).start()
    return jsonify({'status': 'warming up'}), 200

@app.route("/get", methods=["GET", "POST"])
def chat():
    init_chain()
    if request.method == "POST":
        msg = request.form.get("msg") or (request.json.get("msg") if request.is_json else "")
    else:  # GET
        msg = request.args.get("msg", "")
    if not msg:
        return "Please provide a message"
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])  # Return plain text instead of JSON

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)