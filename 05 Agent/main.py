from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from datetime import datetime
import json
import requests
import os
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client (adjust base_url if needed for Gemini)
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Memory store for file content
generated_data = {}

# Tool implementations
def run_command(cmd: str):
    result = os.system(cmd)
    return f"Command executed with exit code {result}"

def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The weather in {city} is {response.text.strip()}."
    return "Something went wrong getting the weather."

def write_file(data: dict):
    filepath = data.get("filepath")
    content = data.get("content")
    if not filepath or not content:
        return "Invalid input. Must provide 'filepath' and 'content'."
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"{filepath} written successfully."

# Mapping tool names to functions
available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
    "write_file": write_file
}

# System prompt
SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who is specialized in resolving user queries.
You work in the following cycle: start → plan → action → observe → output.

For each user query and based on the available tools, you should:
1. Plan the step-by-step approach to solve the query.
2. Select the most appropriate tool.
3. Execute the tool using the correct input.
4. Wait for the output (observation) and then form a final response to the user.

🧠 Rules:
- Follow the Output JSON Format.
- Perform only one step at a time and wait for the next observation before continuing.
- Carefully analyze the user query before taking action.
- Only use tools defined in the tool list below.

📤 Output JSON Format:
{{
    "step": "string",                // One of: "plan", "action", "output", "store"
    "content": "string",             // Explanation or final output
    "function": "string",            // If action, tool name to use
    "input": "object or string"      // If action, input to the function
}}

💾 Global File Strategy:
- For each file (e.g., `Todo/index.html`, `Todo/styles.css`, `Todo/script.js`), generate content separately.
- Store content in memory using a structured JSON message like:
  {{
    "step": "store",
    "filepath": "Todo/index.html",
    "content": "<!DOCTYPE html>..."
  }}
- Then call `write_file` with the same filepath.

🛠️ Available Tools:
- **get_weather**: Takes a city name as an input and returns the current weather.
- **run_command**: Takes a valid Windows terminal command (cmd.exe syntax) as a string and executes it.
- **write_file**: Writes content stored in memory to the disk for a given filepath.

⚠️ Windows Command Guidelines:
- Only use valid **cmd.exe** commands like: `mkdir`, `echo`, `type`, `del`, etc.
- Avoid unsupported Linux-style commands like `ls`, `touch`, `cat`, `rm`, etc.

🚫 Do not respond with natural language messages like "Success" or "Files created".
✅ Always respond ONLY with a valid JSON object following the Output JSON Format.

📌 Termination Rule:
- Once all required files have been created and the task is complete, end with one final response:
  {{
    "step": "output",
    "content": "Project 'Calculator' created successfully. Files: index.html, styles.css, script.js are located in the 'Calculator' directory."
  }}
- Do not repeat the same success message again.
- Do not take any further action after the final output.

  Always respond ONLY with a valid JSON object following the Output JSON Format. Do not return natural language text alone.
"""

# Message history
messages = [
    { "role": "system", "content": SYSTEM_PROMPT }
]

# Main interaction loop
while True:
    query = input("> ")
    if not query.strip():
        continue

    messages.append({ "role": "user", "content": query })

    while True:
        try:
            response = client.chat.completions.create(
                model="gemini-1.5-flash",  # Adjust as needed
                response_format={ "type": "json_object" },
                messages=messages
            )
        except RateLimitError:
            print("⏳ Rate limit reached. Retrying in 60 seconds...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"❌ Unexpected error during OpenAI request: {e}")
            break

        raw_content = response.choices[0].message.content
        print("🧾 Raw content:")
        print(raw_content)
        if not raw_content:
            print("⚠️ Empty response from the model. Skipping this cycle.")
            break

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {e}")
            print("🧾 Raw content was:\n", raw_content)
            break

        parsed_responses = parsed if isinstance(parsed, list) else [parsed]

        for parsed_response in parsed_responses:
            messages.append({ "role": "assistant", "content": json.dumps(parsed_response) })

            step = parsed_response.get("step")

            if step == "plan":
                print(f"🧠: {parsed_response.get('content')}")
                continue

            elif step == "action":
                tool_name = parsed_response.get("function")
                tool_input = parsed_response.get("input")
                print(f"🛠️: Calling Tool: {tool_name} with input {tool_input}")

                if available_tools.get(tool_name):
                    try:
                        output = available_tools[tool_name](tool_input)
                    except Exception as e:
                        output = f"❌ Tool execution error: {e}"

                    messages.append({
                        "role": "user",
                        "content": json.dumps({
                            "step": "observe",
                            "output": output
                        })
                    })
                    break  # ✅ Wait for assistant to reason again on observation

            elif step == "store":
                filepath = parsed_response.get("filepath")
                content = parsed_response.get("content")
                if filepath and content:
                    generated_data[filepath] = content
                    print(f"📄 Stored content for {filepath}")
                continue

            elif step == "output":
                print(f"🤖: {parsed_response.get('content')}")
                break  # ✅ Final output received
