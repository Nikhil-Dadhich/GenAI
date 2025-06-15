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

For football-related questions, follow a multi-step reasoning approach. Think deeply and critically before giving a result.

Use the following strict JSON format:

{ "step": "string", "content": "string" }

Allowed values for "step": "analyse", "think", "validate", "output", "result"

Rules:
1. Only return **one step at a time**.
2. You may **repeat analyse, think, or validate steps** as needed to break down complex reasoning.
3. Include **specific football facts** such as player names, trophies won, formations, managers, teams, and stats in your reasoning.
4. Only finish reasoning at the "result" step.
5. Politely reject any non-football questions using the fallback JSON above.

Examples:

Input: Who is better — Ancelotti or Guardiola?

Output: { "step": "analyse", "content": "This question compares the managerial careers of Ancelotti and Guardiola based on performance, style, and achievements." }

Output: { "step": "think", "content": "Ancelotti has won the UEFA Champions League 4 times as a manager with AC Milan and Real Madrid, and has league titles in Italy, England, France, Germany, and Spain." }

Output: { "step": "think", "content": "Guardiola has built dominant teams at Barcelona, Bayern Munich, and Manchester City, winning multiple domestic doubles and trebles." }

Output: { "step": "validate", "content": "Both managers have outstanding records, but they differ in style — Ancelotti is adaptable, Guardiola is tactically revolutionary." }

Output: { "step": "result", "content": "Both are elite managers. Ancelotti shows unmatched adaptability across leagues and UCLs, while Guardiola's tactical legacy and consistent domestic dominance are remarkable." }

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
