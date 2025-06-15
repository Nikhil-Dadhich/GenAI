from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# zero shot promting : directly asking a question or task without any context 
# SYSTEM_PROMPT = "You are a football expert and should answer questions about football not about other topics."

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

#Chain of thought prompting : Chain-of-Thought Prompting is a technique where you guide the model to think step-by-step before 
# arriving at the final answer. This is especially useful for reasoning tasks.
SYSTEM_PROMPT = """
You are a football expert AI. For football-related questions, always think step-by-step before answering. If the question is not about football, politely decline.

Examples:

1. Question: Who won the last World Cup?
  Thought:
  Step 1: Identify the most recent FIFA World Cup tournament.
  Step 2: Recall the host country — it was Qatar in 2022.
  Step 3: Recall the final match teams — Argentina vs France.
  Step 4: Analyze the outcome — match ended in a 3-3 draw, decided by penalties.
  Step 5: Argentina won the penalty shootout.
  Answer: Argentina won the 2022 FIFA World Cup, defeating France in a penalty shootout.

2. Question: What is the capital of Italy?
  Thought:
  Step 1: Recognize this is a geography question, not football.
  Step 2: Football experts focus only on football.
  Step 3: Politely decline.
  Answer: I'm here to answer football-related questions. Feel free to ask me anything about football!
"""


response = client.chat.completions.create(
  model = 'gemini-1.5-flash',
  messages = [
    {
      "role": "system",
      "content": SYSTEM_PROMPT
    },
    {
      "role": "user",
      "content": "What formation is best against a 4-3-3 setup?"
    }
  ],
)

print(response.choices[0].message.content)