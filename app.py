from pathlib import Path

import gradio as gr

# Make sure all necessary functions are imported from utils
from utils import (create_chatbot_payload, encode_image_to_base64_string,
                   get_environment_settings, load_json_configuration,
                   send_request_to_openai_api)

# Load environment variables and configuration
env_vars = get_environment_settings()
config = load_json_configuration()


def process_image(
    image_path: Path,
    config_section: dict,
    model: str,
    metadata_fields: list = None,
    metadata_extraction_strings: dict = None,
) -> str:
    """Process the image for a specified task using the OpenAI API."""
    image_base64 = encode_image_to_base64_string(image_path)
    if image_base64 is None:
        return "Error: Image encoding failed"

    prompt = config_section.get("prompt", "")

    # Add metadata extraction strings to the prompt if applicable
    if metadata_fields and metadata_extraction_strings:
        metadata_prompts = [
            f"{field}: {metadata_extraction_strings.get(field, '')}"
            for field in metadata_fields
        ]
        prompt += "\n" + "\n".join(metadata_prompts)

    payload = create_chatbot_payload(
        image_base64,
        prompt,
        config_section.get("addendum", ""),
        config_section.get("output", ""),
        env_vars["models"][model],
        env_vars["max_tokens"],
    )
    return send_request_to_openai_api(payload, env_vars["api_key"], env_vars["api_url"])


# TODO: consider handling each task asynchronosly so that we can update the UI as each task completes
def main(
    language: str, document_path: str, metadata_to_extract: list, selected_model: str
) -> tuple:
    """Main function to process the document image and extract required information."""
    document_path = Path(document_path)
    language_config = config[language]

    classification = process_image(
        document_path,
        language_config["classification_prompts"],
        selected_model,
    )

    metadata_config = language_config["metadata_prompts"].get(classification, {})
    metadata = process_image(
        document_path,
        metadata_config,
        selected_model,
        metadata_to_extract,
        language_config["metadata_extraction_string"],
    )

    tests_config = language_config["tests_prompts"]
    tests = (
        process_image(document_path, tests_config, selected_model)
        if classification in ["COA", "SCD+COA"]
        else "No tests to extract"
    )

    return classification, metadata, tests


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            document_input = gr.Image(type="filepath", label="Upload Document Image")
            model_selector = gr.Dropdown(list(env_vars["models"].keys()), label="Model")
            language_selector = gr.Dropdown(["it", "en"], label="Document Language")
            metadata_to_extract_selector = gr.CheckboxGroup(
                config["metadata_to_extract"], label="Metadata to Extract"
            )

        with gr.Row():
            classification_display = gr.Textbox(label="Classification")
            metadata_display = gr.Textbox(label="Extracted Metadata")
            tests_display = gr.Textbox(label="Extracted Tests", max_lines=10)

        analyze_button = gr.Button("Analyze")
        analyze_button.click(
            fn=main,
            inputs=[
                language_selector,
                document_input,
                metadata_to_extract_selector,
                model_selector,
            ],
            outputs=[classification_display, metadata_display, tests_display],
        )
        demo.launch(share=True)
