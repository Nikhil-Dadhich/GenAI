from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
  model="models/embedding-001",
  google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="VectorStore",
    embedding=embedding_model,
)

# Take User Query
query = input("> ")

# Vector Similarity Search [query] in DB
search_results = vector_db.max_marginal_relevance_search(
    query=query,
    k=5,             # Final top 5 chunks to return
    fetch_k=20,      # Consider top 20 closest chunks before filtering
    lambda_mult=0.7  # 0.7 = prioritize relevance with some diversity
)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""
You are a helpful and precise AI assistant.

You must answer the user's query using **only** the provided context extracted from a PDF document. Each chunk of context contains:
- The page content
- The page number
- The file location

### Instructions:
- Base your answers strictly on the given context.
- If the answer is partially available, mention which page the user should refer to for more details.
- If the context does **not** contain enough information to answer the query, say: "I couldn't find relevant information in the current document context. Please refer to the document manually."
- Rephrase the user's query to ensure clarity and precision in your response.
- Rephrase the answer to ensure it is clear.
- Try to give depth in your answers, but do not fabricate information.
- If the user asks for a file location, provide the full path to the file.
### Context:
{context}
"""


chat_completion = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": query },
    ]
)

print(f"🤖: {chat_completion.choices[0].message.content}")
