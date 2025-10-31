import os
import openai
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4.1-mini'


if __name__ == "__main__":
    prompt = "How to implement a humanlike reasoning model using small scale LLMs?"
    client = OpenAI()
    print("\nGetting response from LLM...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
    )
    text = response.choices[0].message.content
    time.sleep(5)
    print(f"Response from LLM: {text}")