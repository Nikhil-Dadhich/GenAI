from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# zero shot promting
SYSTEM_PROMPT = "You are a football expert and should answer questions about football not about other topics."
response = client.chat.completions.create(
  model = 'gemini-1.5-flash',
  messages = [
    {
      "role": "system",
      "content": SYSTEM_PROMPT
    },
    {
      "role": "user",
      "content": "Captial of Spain?"
    }
  ],
)

print(response.choices[0].message.content)