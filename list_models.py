import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env if you're using one
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment.")

genai.configure(api_key=api_key)

print("Available models that support generateContent:\n")

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"Model name: {model.name}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        print("-" * 60)