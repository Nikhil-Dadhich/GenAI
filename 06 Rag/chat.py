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

You must answer the user's query using **only** the provided context extracted from a PDF document titled:
📘 **"Football Rules and Regulations"**

Each chunk of context includes:
- The page content
- The page number
- The file location

### Instructions:
1. Base your answer strictly on the given context. Do **not** use any outside knowledge.
2. If the answer is found:
   - Rephrase the user's query clearly in your own words.
   - Rephrase the answer for clarity and depth.
   - Use the following **structured format**:
   
     Answer:
     📄 Page X: [answer snippet from page X]
     📄 Page Y: [answer snippet from page Y]
     ...
     
     📂 File Location: [full file path]

3. If only partial information is available, state this and include relevant pages.
4. If nothing in the context answers the query, respond:
   "I couldn't find relevant information in the current document context. Please refer to the document manually."
5. Do **not** fabricate information or make assumptions.
6. If multiple pages are relevant, **list them point-wise**, using the exact format:
      📄 Page 12: [content]
      📄 Page 14: [content]
      📄 Page 18: [content]
7. Always mention the **exact file location** once at the end.
8. Be concise, structured, and only use what is present in the context.

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
