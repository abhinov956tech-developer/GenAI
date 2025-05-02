import json
from dotenv import load_dotenv
import google.generativeai as genai
import os
import time
import requests
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_weather(city: str):
    print("🔨 Tool Called: get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


def get_book():
    response = requests.get('https://potterapi-fedeperin.vercel.app/en/books')
    books = response.json()
    return books


Available_Tools= {
"get_weather":{
"fn": get_weather,
"description":"Takes a city name as an input and returns the current weather for the city"
},
"get_books":{
"fn": get_book,
"description":"Takes a user query regarding Harry potter characters or any stories plot and returns with a answer"
}
}

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
    - Always include all required fields in your JSON responses.
    - For action steps, always include both "function" and "input" fields.
    - Finish with an output step that summarizes the result for the user in natural language.

    Output JSON Format:
    {{
    "step": "string",
    "content": "string",
    "function": "The name of function if the step is action",
    "input": "The input parameter for the function"
    }}

     Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput
    - get_book: Takes a user query regarding Harry potter characters or any stories plot and returns with a answer
    
    Example:
    User Query: What is the weather of New York?
    Output: { "step": "plan", "content": "The user is interested in weather data of New York" }
    Output: { "step": "plan", "content": "From the available tools I should call get_weather" }
    Output: { "step": "action", "function": "get_weather", "input": "New York" }
    Output: { "step": "observe", "output": "12 Degree Celsius" }
    Output: { "step": "output", "content": "The weather for New York seems to be 12 degrees Celsius." }
    Output: { "step": "done", "content": "Done" }

    Example 2:
     User Query: What is the name of the owl that harry potter had?
    Output: {{ "step": "plan", "content": "The user is curious for knowing the name of the harry potter owl " }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_book" }}
    Output: {{ "step": "action", "function": "get_book" }}
    Output: {{ "step": "observe", "output": "Hedwig" }}
    Output: {{ "step": "output", "content": "The name of Harry potter owl is Hedwig" }}

    """,
 generation_config={
        "response_mime_type": "application/json"
    },
 
)

history =[]

while True:
    user_query=input('>')       
    history.append({"role": "user", "parts": [{"text": user_query}]})
    chat = model.start_chat(history=history)
    if user_query.lower() in ['exit', 'quit']:
          break  

    while True:
        
            try:
            
                response = chat.send_message("Continue")
                # print(response.text)
                
                try:
                    parsed_output = json.loads(response.text)
                    
                
                    if parsed_output.get("step") == "plan":
                        print(f"🧠: {parsed_output.get('content')}")
                    
                    elif parsed_output.get("step") == "action":
                        tool_name = parsed_output.get("function")
                        tool_input = parsed_output.get("input")
                        
                        if tool_name in Available_Tools:
                            output = Available_Tools[tool_name]["fn"](tool_input)
                            observe_response = {"step": "observe", "output": output}
                            print(f"🔍: Observation - {output}")
                            
                            
                            response = chat.send_message(json.dumps(observe_response))
                            
                            try:
                                final_parsed = json.loads(response.text)
                                if final_parsed.get("step") == "output":
                                    print(f"🤖: {final_parsed.get('content')}")
                            except json.JSONDecodeError:
                                print("Error parsing response")
                                
                    elif parsed_output.get("step") == "output": 
                        print(f"🤖: {parsed_output.get('content')}")
                        
                    elif parsed_output.get("step") == "done":
                        break
                        
                except json.JSONDecodeError:
                    print("Error: Could not parse response as JSON")
                    break
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                if "429" in str(e) or "quota" in str(e).lower():
                    print("Rate limit hit. Waiting 60 seconds before trying again...")
                    time.sleep(60)  
                    continue
                break

           
   
            
            
       
    

