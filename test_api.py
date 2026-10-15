from google import genai
import os
from dotenv import load_dotenv

# Load your new AQ API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# The new SDK uses a Client object instead of configure()
client = genai.Client(api_key=api_key)

try:
    print("Sending request to Gemini using the new SDK...")
    # Using the current standard model
    response = client.models.generate_content(
        model='gemini-2.5-flash', # Updated model name for the new SDK
        contents="Say 'Hello, my new API key works!'"
    )
    print("\nSUCCESS! AI says:", response.text)
except Exception as e:
    print("\nFAILED! Error:", str(e))