"""
OpenRouter API Integration
Part 4: AI API integration for LLM responses
"""
import httpx
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def call_openrouter(prompt: str, model: str = "openai/gpt-oss-20b:free") -> str:
    """Call OpenRouter API for LLM responses"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        return "OpenRouter API key not configured"
    
    try:
        with httpx.Client() as client:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenRouter API error: {str(e)}"

def is_openrouter_configured() -> bool:
    """Check if OpenRouter is configured"""
    return bool(os.getenv("OPENROUTER_API_KEY"))