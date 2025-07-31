from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document 


# Extract text from PDF files
def load_pdf_files(Data):
    loader = DirectoryLoader(Data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents



def filter_to_minimal_docs(docs : List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content= doc.page_content,
                metadata={"source": src}

            )
        )
    return minimal_docs



#Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk


# Downloading embeddings from HuggingFace

from langchain.embeddings import HuggingFaceEmbeddings
def download_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embeddings
embeddings = download_embeddings()


#Adding the Document creator information


creater = Document(
    page_content= "The application you are seeing is developed/created by Sai Charan, He is " \
    "He is a Data Scientist and Generative AI Developer.",
    metadata={"source": "creator_info"}
)