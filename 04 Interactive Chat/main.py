from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client (using Gemini endpoint)
client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Chain-of-Thought prompt for football expert
SYSTEM_PROMPT = """
You are a football expert AI. You must only answer football-related questions. If a question is not related to football (soccer), respond with the following JSON:

{ "step": "result", "content": "I'm specialized in football (soccer). Please ask a football-related question." }

For football-related questions, reason step-by-step using the following format:

{ "step": "string", "content": "string" }

Allowed values for "step": "analyse", "think", "output", "validate", "result"

Rules:
1. Only return one step at a time.
2. Wait for user input before continuing.
3. End reasoning at the "result" step.
4. Always stick to football — no math, science, or general knowledge.

Example:
Input: What is the best counter formation to 4-3-3?
Output: { "step": "analyse", "content": "This is a tactical football question asking how to counter a 4-3-3 formation." }

Example:
Input: What is the capital of France?
Output: { "step": "result", "content": "I'm specialized in football (soccer). Please ask a football-related question." }

"""

# Start conversation
messages = [{ "role": "system", "content": SYSTEM_PROMPT }]
user_input = input("⚽ Enter your football question: ")
messages.append({ "role": "user", "content": user_input })

while True:
    response = client.chat.completions.create(
        model='gemini-1.5-flash',
        response_format={"type": "json_object"},
        messages=messages
    )

    content = response.choices[0].message.content
    messages.append({ "role": "assistant", "content": content })

    try:
        step_obj = json.loads(content)
        step = step_obj.get("step")
        step_text = step_obj.get("content")
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON format from model. Raw response:\n", content)
        break

    # Display step output
    print(f"\n🧠 Step: {step}\n📄 {step_text}")

    # Break loop if final step reached
    if step == "result":
        print("\n✅ Final answer complete.")
        break

    # Wait for user to continue
    # input("➡️ Press Enter to continue to next step...")
