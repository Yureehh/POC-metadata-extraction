import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional

import gradio as gr
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.getenv("MODEL", "gpt-4-vision-preview")
MODEL2 = os.getenv("MODEL2", "gpt-4-1106-preview")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
TEMPERATURE = int(os.getenv("TEMPERATURE", 0))
METADATA_OPTIONS = [
    "IdLotto",
    "IdLottoMadre",
    "CodiceMateriaPrima",
    "DDT",
    "DataDDT",
    "DataConsegna",
    "PresenzaConformità",
]

# Verify if API key is available
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


def encode_image_to_base64(image_path: Path) -> Optional[str]:
    """Encodes the image at the given path to base64."""
    try:
        with image_path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError as e:
        print(f"Image file not found: {image_path}")
        return None


def build_payload(
    base64_image: str, prompt: str, addendum: str, output: str, model: str = MODEL
) -> Dict:
    """Builds the payload for the API request."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": addendum}]},
            {"role": "user", "content": [{"type": "text", "text": output}]},
        ],
        "max_tokens": MAX_TOKENS,
    }


def build_payload2(metadata: str, prompt: str, model: str = MODEL2) -> Dict:
    """Builds the payload for the API request. No image is passed."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "text", "text": metadata}]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
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


def process_image(image_path: Path, config: Dict, task_type: str, **kwargs) -> str:
    """Processes the image for a specified task."""
    base64_image = encode_image_to_base64(image_path)
    if base64_image is None:
        return "Error: Image encoding failed"

    prompt = config.get(task_type, {}).get("prompt", "")
    addendum = config.get(task_type, {}).get("addendum", "")
    output = config.get(task_type, {}).get("output", "")
    metadata_extraction_string = config.get("metadata_extraction_string", {})

    if metadata_to_extract := kwargs.get("metadata_to_extract"):
        prompt_addition = "".join(
            f"{metadata}: [{metadata_extraction_string[metadata]}]\n"
            for metadata in metadata_to_extract
        )
        prompt += f"{prompt_addition}."

    payload = build_payload(base64_image, prompt, addendum, output)
    return post_request_to_api(payload)


def process_metadata_with_llm(config: Dict, metadata: str, model: str = MODEL2) -> Dict:
    """Processes the metadata and returns a dictionary."""
    prompt = config.get("metadata", {}).get("prompt", "")
    payload = build_payload2(metadata, prompt, model=model)
    return post_request_to_api(payload)


def process_metadata(metadata: str) -> Optional[Dict]:
    """Attempts to parse JSON metadata and returns a dictionary if successful."""
    try:
        return "\n".join(
            f"{key}: {value}" for key, value in json.loads(metadata).items()
        )
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return "An error occurred while parsing the metadata"


def main(language: str, document_path: str, metadata_to_extract: str) -> str:
    try:
        document_path = Path(document_path)
        config_path = Path("config/conf.json")
        config = load_json_config(config_path)
        class_config = config["classification_prompts"][language]
        metadata_config = config["metadata_prompts"][language]

        classification = process_image(document_path, class_config, "classification")
        metadata = process_image(
            document_path,
            metadata_config,
            classification,
            metadata_to_extract=metadata_to_extract,
        )
        parsed_metadata = process_metadata(metadata)

        return classification, parsed_metadata

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            document_path = gr.Image(type="filepath", label="Image")
            with gr.Column():
                model = gr.Dropdown(
                    [MODEL], value=MODEL, label="Model", info="Which model to use?"
                )
                language = gr.Dropdown(
                    ["it", "en"],
                    label="Language",
                    info="Which language is the document in?",
                )
                metadata_to_extract = gr.CheckboxGroup(
                    METADATA_OPTIONS,
                    value=METADATA_OPTIONS,
                    label="Metadata",
                    info="What metadata to extract?",
                )
        with gr.Row():
            classification = gr.Textbox(
                label="Classification", info="The document has been classified as:"
            )
            metadata_extracted = gr.Textbox(
                label="Metadata", info="The following metadata have been extracted:"
            )
        query_btn = gr.Button("Ask")
        query_btn.click(
            fn=main,
            inputs=[language, document_path, metadata_to_extract],
            outputs=[classification, metadata_extracted],
        )

    demo.launch(share=True)
