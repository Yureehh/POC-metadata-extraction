
# Project Overview

This proof-of-concept (POC) project demonstrates the application of GPT-4 Vision for classifying and extracting metadata from client documents. The goal is to assess the feasibility of automating document processing to improve efficiency and accuracy in data management.

## Features

- **Document Classification**: Uses GPT-4 Vision for document type classification.
- **Metadata Extraction**: Automates the extraction of important metadata from documents.
- **User Interface**: Integrates a Gradio web interface for easy interaction and processing.
- **Error Handling**: Implements robust error handling mechanisms for consistent performance.

## Technical Components

- **GPT-4 Vision**: Employs OpenAI's advanced AI model for processing and understanding document images.
- **Python**: Backend developed in Python, offering a wide range of library support.
- **Gradio**: Provides an interactive web platform for document upload and display of results.
- **Custom Utilities**: Includes utility functions for handling API requests, image encoding, and payload management.

## Disclaimer

- The prompts and configurations used in this project are tailored specifically for our client's requirements. They should be adapted to fit the unique needs and data formats of other users or use cases.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone [repository-url]
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY=[your-key]
   export HF_API_KEY=[your-key]
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

## Usage

1. **Launch the Application**:
   - Execute `app.py` to start the Gradio web interface.

2. **Using the Interface**:
   - Upload the document image through the Gradio interface.
   - Select the appropriate model, specify the document's language, and choose the metadata fields you wish to extract.
   - Click the "Analyze" button to process the document.

3. **View Results**:
   - The application will display the classification of the document, the extracted metadata, and additional test results (if applicable).

## Contributions

We welcome contributions to this project, especially as it is in the POC stage. Feel free to fork the repository, make your changes, and submit pull requests with your improvements.