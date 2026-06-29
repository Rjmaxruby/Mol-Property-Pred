from config import GROQ_API_KEY
from groq import Groq

client = None

if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)


def generate(prompt: str) -> str:

    if client is None:
        raise RuntimeError("GROQ_API_KEY not found.")

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.3,
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()