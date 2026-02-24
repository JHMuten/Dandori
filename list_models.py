from dotenv import load_dotenv
import os
from google import genai

# Load .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

print("Available models:\n")

for model in client.models.list():
    print(f"Model: {model.name}")
    print(f"  Supported methods: {model.supported_generation_methods}")
    print("-" * 60)
    