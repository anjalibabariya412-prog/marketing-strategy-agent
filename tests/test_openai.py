from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

client = OpenAI(api_key=api_key)

print("Checking models available to this API key...\n")

try:
    models = client.models.list()

    for model in models.data:
        print(model.id)

except Exception as e:
    print("Failed to access OpenAI API:")
    print(e)