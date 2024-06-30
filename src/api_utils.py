# src/api_utils.py
import aiohttp
import asyncio
import functools
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def cache(func):
    cache_dict = {}
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache_dict:
            cache_dict[key] = await func(*args, **kwargs)
        return cache_dict[key]
    return wrapper

@cache
async def get_response(prompt, temperature, top_p, presence_penalty, frequency_penalty, n):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=n,
            max_tokens=1000
        )
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"Error in API call: {e}")
        return []
