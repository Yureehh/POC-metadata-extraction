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
    image_path: Path, config_section: dict, model: str
) -> str:
    """Process the image for a specified task using the OpenAI API."""
    image_base64 = encode_image_to_base64_string(image_path)
    if image_base64 is None:
        return "Error: Image encoding failed"
    payload = create_chatbot_payload(
        image_base64,
        config_section.get("prompt", ""),
        config_section.get("addendum", ""),
        config_section.get("output", ""),
        env_vars["models"][model],
        env_vars["max_tokens"],
    )
    return send_request_to_openai_api(payload, env_vars["api_key"], env_vars["api_url"])

#TODO: consider handling each task asynchronosly so that we can update the UI as each task completes
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
    metadata = process_image(
        document_path,
        language_config["metadata_prompts"][classification],
        selected_model,
    )
    tests = (
        process_image(
            document_path, language_config["tests_prompts"], selected_model
        )
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
            metadata_to_extract = gr.CheckboxGroup(
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
                metadata_to_extract,
                model_selector,
            ],
            outputs=[classification_display, metadata_display, tests_display],
        )
        demo.launch(share=True)
