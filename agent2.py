import json
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langsmith import Client
import google.generativeai as genai
import os
import time
import requests
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

langsmith_client=Client()

def trace_gemini_call(model_name, prompt, response_text):
    """Trace a single Gemini API call in LangSmith"""
    langsmith_client.create_run(
        name="Gemini Generate",
        run_type="llm",
        inputs={"model": model_name, "prompt": prompt},
        outputs={"response": response_text},
        tags=["gemini"]
    )

  
@traceable
def get_weather(city: str):
    print("🔨 Tool Called: get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"

#def run_command(command: str):
    #result=os.system(command)
   # return f"Command executed with result code: {result}"
@traceable
def run_command(command: str):
    try:
        
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            output = result.stdout if result.stdout else "Command executed successfully"
            return output
        else:
            return f"Error (code {result.returncode}): {result.stderr}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"


Available_Tools= {
"get_weather":{
"fn": get_weather,
"description":"Takes a city name as an input and returns the current weather for the city"
},

"run_command":{
    "fn":run_command,
    "description":"Takes a command as input to execute on system and returns ouput"
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
    -  Follow the Output JSON Format strictly.
    -  Always perform one step at a time and wait for next input.
    -  Carefully analyze the user query.
    -  Always include all required fields in your JSON responses.
    -  For action steps, always include both "function" and "input" fields.
    -  Finish with an output step that summarizes the result for the user in natural language.

    Output JSON Format:
    {{
     "step": "string",  // "plan", "action", "output", or "done"
    "content": "string", // Description or output message
    "function": "string", // Required for "action" step only - name of the function to call
    "input": "string" // Required for "action" step only - parameter to pass to the function
    }}

     Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput
    
    
    Example 1:
    User Query: What is the weather of New York?
    Output: { "step": "plan", "content": "The user is interested in weather data of New York" }
    Output: { "step": "plan", "content": "From the available tools I should call get_weather" }
    Output: { "step": "action", "function": "get_weather", "input": "New York" }
    Output: { "step": "observe", "output": "12 Degree Celsius" }
    Output: { "step": "output", "content": "The weather for New York seems to be 12 degrees Celsius." }
    Output: { "step": "done", "content": "Done" }

    Example 2:
     User Query: Create a python file named hello.py in the Langgraph folder
    Output: { "step": "plan", "content": "I need to create a python file named hello.py in the Langgraph folder" }
    Output: { "step": "plan", "content": "From the available tools I will use run_command to execute shell commands" }
    Output: { "step": "action", "function": "run_command", "input": "mkdir -p Langgraph && touch Langgraph/hello.py" }
    Output: { "step": "observe", "output": "Command executed with result code: 0" }
    Output: { "step": "output", "content": "The python file hello.py has been created in the Langgraph folder." }
    Output: { "step": "done", "content": "Done" }
    """,
 generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.2
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

           
   
            
            
       
    

