from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

text = "hala madrid e nada mais"

response = client.embeddings.create(
  input=text,
  model="text-embedding-004"
)

print(response.data[0].embedding)
print({len(response.data[0].embedding)})
