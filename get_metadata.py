import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_keywords(chunk):
    try:
        response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Use the desired GPT model
        messages=[
            {
                 "role": "system",
                 "content": "You are a tool that extracts concise keywords from a given text. Respond only with a plain list of keywords, each separated by commas."
                },
                {
                 "role": "user",
                 "content": f"Extract keywords from this text chunk:\n{chunk}"
                }
            ],
            temperature=0.5
        )
        keywords = response.choices[0].message.content.strip()
        keywords_list = [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]
        return keywords_list
    except Exception as e:
        print(f"Error generating keywords: {e}")
        return []