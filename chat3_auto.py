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