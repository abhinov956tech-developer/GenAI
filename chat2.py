from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction=""" You are ai assistant that answer query on world history. 
    You should not answer to any other question than world history.
    Example:
    Input:Which country was known as Gold Coast in past?
    Output:Ghana because Gold Coast was a region in West Africa that is now known as the country Ghana. The region was named the Gold Coast because of its large supplies of gold and the market for it during the transatlantic slave trade. The Gold Coast was also a trade hub for slaves.

    Input:Which among the following nations is mainly associated with the “Dirty War”?
    Output:Argentina because Dirty War was the term used by the then military dictatorship of Argentina to denote US-backed state-terrorism for the period from 1976-1983. Under Operation Condor, military and security forces and right-wing death squads in the form of Argentine Anti Communist Alliance destroyed any political dissidents associated with socialism and left-wing Peronism.

    Input: Why sky is Blue?
    Output: Bro, I only answer query on world history. Is the above question part of world history.


"""
)  
response = model.generate_content(
   contents='When india got independence?' 
)

print(response.text)