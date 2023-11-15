import os
import json
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict
import gradio as gr 

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.getenv("MODEL", "gpt-4-vision-preview")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
TEMPERATURE = int(os.getenv("TEMPERATURE", 0))

if not API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable")

def load_json_config(file_path: Path) -> Dict:
    """Loads and returns JSON configuration from the given file path."""
    try:
        with file_path.open("r") as file:
            return json.load(file)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {file_path}")
        raise e

def encode_image_to_base64(image_path: Path) -> str:
    """Encodes the image at the given path to base64."""
    try:
        with image_path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError as e:
        print(f"Image file not found: {image_path}")
        raise e

def build_payload(base64_image: str, prompt: str, addendum: str, output: str) -> Dict:
    """Builds the payload for the API request."""
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}]},
            {"role": "user", "content": [{"type": "text", "text": addendum}]},
            {"role": "system", "content": [{"type": "text", "text": output}]},
        ],
        "max_tokens": MAX_TOKENS
    }

def post_request_to_api(payload: Dict) -> Dict:
    """Posts a request to the API and returns the response."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"API Request Failed: {e}")
        raise

def process_image(image_path: Path, config: Dict, task_type: str) -> str:
    """Processes the image for a specified task."""
    base64_image = encode_image_to_base64(image_path)
    prompt = config.get(task_type, {}).get("prompt", "")
    addendum = config.get(task_type, {}).get("addendum", "")
    output = config.get(task_type, {}).get("output", "")
    payload = build_payload(base64_image, prompt, addendum, output)
    return post_request_to_api(payload)

def main(language: str, document_path: str):
    try:
        document_path = Path(document_path)
        config_path = Path("config/conf.json")
        config = load_json_config(config_path)
        class_config = config["classification_prompts"][language]
        metadata_config = config["metadata_prompts"][language]

        classification = process_image(document_path, class_config, "classification")
        metadata = process_image(document_path, metadata_config, classification)

        return classification, metadata

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    with gr.Blocks() as demo:
        language = gr.Dropdown(["it", "en"], label="Language")
        document_path = gr.Image(type="filepath", label="Image")
        classification = gr.Textbox(label="Classification")
        metadata = gr.Textbox(label="Metadata")
        query_btn = gr.Button("Ask")
        query_btn.click(fn=main, inputs=[language, document_path], outputs=[classification, metadata])

    demo.launch(share=True)