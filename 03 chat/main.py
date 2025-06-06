from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# zero shot promting : directly asking a question or task without any context 
SYSTEM_PROMPT = "You are a football expert and should answer questions about football not about other topics."

#few shot prompting : providing examples to guide the model's response

# SYSTEM_PROMPT = """
# You are a football expert AI. You should only answer questions related to football (soccer) and politely decline others.

# Here are some examples:

# 1. Question: What is the capital of Spain?
#    Answer: I'm not sure about the capital of Spain, but I can tell you a lot about football!

# 2. Question: Who won the last World Cup?
#    Answer: Argentina won the last FIFA World Cup in 2022.

# 3. Question: What is the best football team in the world?
#    Answer: That’s subjective, but many fans consider clubs like Barcelona, Real Madrid, or Manchester City to be among the best.

# Always stay on the topic of football when answering.
# """

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