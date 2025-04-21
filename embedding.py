from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
text ="Taj Mahal is in India and is a symbol of love."

response=genai.embed_content(
    model="models/embedding-001",
    content=text
)

print("Vector embeddings", response["embedding"])