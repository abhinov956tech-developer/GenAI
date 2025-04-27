from dotenv import load_dotenv
import google.generativeai as genai
import os
import json

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction="""You are AI assistant that solves complex problem by breaking down the problem step by step.
    For the given user input, analyse the input and break down the problem step by step.
    Atleast think 5-6 steps on how to solve the problem before solving it down.

    The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

    Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

    Rules:
    1. Follow the strict JSON output as per Output schema.
    2. Always perform one step at a time and wait for next input
    3. Carefully analyse the user query

    Output Format:
    {{ step: "string", content: "string" }}

    Example:
    Input: What is 2 + 2.
    Output: {{ step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }}
    Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
    Output: {{ step: "output", content: "4" }}
    Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
    Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}
""",
    generation_config={
        "response_mime_type": "application/json"
    }
)

history = [
   {"role": "user", "parts": ["what is 3 + 4 * 5"]},
    {"role": "model", "parts": [json.dumps({"step": "think", "content": "Following the order of operations (PEMDAS/BODMAS), multiplication should be performed before addition.  Therefore, the calculation should be 4 * 5 first, and then the result added to 3."})]},
]

chat = model.start_chat(history=history)


response = chat.send_message("Please continue to the next step.")

print(response.text)