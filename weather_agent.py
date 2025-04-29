import json
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_weather(city: str):
    return "40 degree celcius"

user-query=input('>')

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction= """You are an AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput
    
    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}
""",
 generation_config={
        "response_mime_type": "application/json"
    },
 
)

history =[
         
          {"role": "user", "parts": ["What is the current weather of Guwahati"]},
          {"role": "model", "parts":[json.dumps({"step": "plan", "content": "The user is interested in weather data of Guwahati"})]},
          {"role": "model", "parts": [json.dumps({"step": "plan", "content": "From the available tools I should call get_weather"})]},
          {"role": "model", "parts": [json.dumps({"step": "action", "function": "get_weather", "input": "Guwahati"})]},
          {"role": "model", "parts": [json.dumps({"step": "observe", "output": "40 degree celcius"})]}
   ] 
 



chat=model.start_chat(history=history)
  
response=chat.send_message("Continue")

print(response.text)