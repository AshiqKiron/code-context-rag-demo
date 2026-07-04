from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from data.mock_codebase import CODE_SNIPPETS

def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_text(CODE_SNIPPETS)
    db = FAISS.from_texts(texts, embeddings)
    return db

def get_relevant_context(db, query):
    docs = db.similarity_search(query)
    return "\n".join([doc.page_content for doc in docs])