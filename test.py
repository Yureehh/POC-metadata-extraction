import os
import json
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

# Constants
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4-vision-preview"
MAX_TOKENS = 1024
TEMPERATURE = 0

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable")

def load_json_config(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")

def build_payload(base64_image, prompt, double_check):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}]},
            {"role": "system", "content": [{"type": "text", "text": double_check}]}
        ],
        "max_tokens": MAX_TOKENS
    }

def classify_document(image_path, class_config):
    base64_image = encode_image_to_base64(image_path)
    prompt = class_config["prompt"]
    double_check = class_config["double_check"]
    payload = build_payload(base64_image, prompt, double_check)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise ConnectionError(f"API Request Failed: Status Code {response.status_code}, Response {response.text}")

if __name__ == "__main__":
    config_path = os.path.join("config", "classification.json")
    config = load_json_config(config_path)
    language = "it"
    class_config = config["classification_prompts"][language]
    document_path = os.path.join("docs", "01 - Dichiarazione di Conformità Lotto L 254SC selez_page1.png")

    try:
        classification = classify_document(document_path, class_config)
        print(classification)

    except Exception as e:
        print(f"An error occurred: {e}")