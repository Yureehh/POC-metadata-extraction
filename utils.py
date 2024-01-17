import base64
import json
from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Dict, Optional

import requests

# Load environment variables
load_dotenv()

def get_environment_settings():
    if api_key := os.getenv("OPENAI_API_KEY"):
        return {
            "api_key": api_key,
            "api_url": "https://api.openai.com/v1/chat/completions",
            "models": {
                "gpt-4-vision": os.getenv("MODEL", "gpt-4-vision-preview"),
                "gpt-4-1106": os.getenv("MODEL2", "gpt-4-1106-preview"),
            },
            "max_tokens": int(os.getenv("MAX_TOKENS", 1024)),
            "temperature": float(os.getenv("TEMPERATURE", 0.0)),  # Changed to float
            "default_model": "gpt-4-vision",
        }
    else:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable")


def load_json_configuration():
    config_path = Path("config/config.json")
    if not config_path.exists():
        raise FileNotFoundError("Configuration file not found at 'config/config.json'")
    with config_path.open("r") as file:
        return json.load(file)


def encode_image_to_base64_string(image_path: Path) -> Optional[str]:
    try:
        with image_path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        return None


def create_chatbot_payload(
    image_base64: str,
    prompt: str,
    addendum: str,
    output: str,
    model: str,
    max_tokens: int,
) -> Dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": addendum}]},
            {"role": "user", "content": [{"type": "text", "text": output}]},
        ],
        "max_tokens": max_tokens,
    }


def create_text_payload(
    text: str, prompt: str, model: str, max_tokens: int, temperature: int
) -> Dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def send_request_to_openai_api(payload: Dict, api_key: str, api_url: str) -> Dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"API Request Failed: {e}")
        raise
