from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os
load_dotenv()


pdf_path = Path(__file__).parent / "football_rules.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

#Chunking

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=400
)

split_docs = text_splitter.split_documents(documents=docs)

#Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
  model="models/embedding-001",
  google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_store = QdrantVectorStore.from_documents(
  documents=split_docs,
  url="http://localhost:6333",
  collection_name="VectorStore",
  embedding=embedding_model,
)

print("Indexing complete. You can now query the vector store.")