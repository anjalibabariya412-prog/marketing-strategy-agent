import os
import sys

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from groq import Groq
from backend.app.core.config import settings

def test_structured_output():
    """
    Temporary throwaway test to verify Groq's JSON mode structured output capabilities.
    """
    try:
        client = Groq(api_key=settings.groq_api_key)

        system_prompt = (
            "You are a marketing strategy assistant. Respond ONLY with valid JSON conforming to this schema:\n"
            "{\n"
            '  "missing_field": "<short field name>",\n'
            '  "reason": "<why this matters>"\n'
            "}\n"
            "Do not include any conversational text or markdown formatting outside the JSON object."
        )

        user_prompt = (
            "A bakery sells cupcakes and wants more customers. "
            "Identify one missing piece of information that would help create a marketing strategy."
        )

        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content
        print("Raw response from LLM:")
        print(raw_text)
        print("-" * 50)

        # Parse raw JSON text into Python dictionary
        try:
            parsed_json = json.loads(raw_text)
            print("Successfully parsed JSON dictionary:")
            print(f"Type: {type(parsed_json)}")
            print(f"Content: {json.dumps(parsed_json, indent=2)}")
            print("-" * 50)
            print(f"missing_field: {parsed_json.get('missing_field')}")
            print(f"reason: {parsed_json.get('reason')}")
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: Failed to parse LLM output as JSON. Details: {e}")

    except Exception as e:
        print(f"Error during LLM call: {e}")

if __name__ == "__main__":
    test_structured_output()
