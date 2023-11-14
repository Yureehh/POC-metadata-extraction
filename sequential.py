import os
import io
import base64
from operator import itemgetter
from typing import Literal

from PIL import Image
import openai
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(temperature=0.0, model="gpt-4-vision-preview", max_tokens=1024)


def get_api_key():
    try:
        return os.getenv("OPENAI_API_KEY")
    except:
        raise Exception("Missing OPENAI_API_KEY environment variable")


doc_type_prompt = PromptTemplate.from_template(
    """
    Classify the document passed in the input as one of "SCD", "COA", "DDT" or "Other".
    SCD stands for "Specification Control Document"
    COA stands for "Certificate of Analysis"
    DDT stands for "Document of Trasport"

    Other is whatever document that is not a SCD, a COA or a DDT.
    The document is passed as a png file containing data either in English or Italian.
    The way to classify the document is to read it and understand what type of document it is
    according to the information contained in it. I expect you to be able to classify the document
    even if it is in a language you don't understand since the classification is based on the
    information contained in the document and not on the language used to express it.
    Moreover I expect you to know what a SCD, a COA and a DDT are.

    Image: {img}
    Document Type: This is the classified doc type of the document"""
)

metadata_prompt = PromptTemplate.from_template(
    """

    In particular the metadata you are able to extract are:
    - id_lotto: the id of the lotto
    - id_lotto_madre: the id of the mother lot
    - codice_materia_prima: the code of the raw material
    - ddt: the ddt number
    - data_ddt: the date of the ddt
    - data_consegna: the delivery date

    Document Type: {doc_type}
    Those are the metadata extracted from the document:
    """
)


def encode_image_to_base64(image_path, output_size=(256, 256)):
    """
    Resize an image and encode it to a base64 string.

    :param image_path: Path to the image file.
    :param output_size: New size for the image as a tuple (width, height).
    :return: Base64 encoded string of the resized image.
    """
    # Open the image
    with Image.open(image_path) as img:
        # # Resize the image
        # img = img.resize(output_size)

        # Save the image to a bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode the image to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    #store image
    with open("imageToSave.png", "wb") as fh:
       fh.write(base64.decodebytes(encoded_image.encode()))
    return encoded_image


image = encode_image_to_base64(
    r"docs/01 - Dichiarazione di Conformità Lotto L 254SC selez_page1.png"
)

doc_type_chain = doc_type_prompt | llm | StrOutputParser()
metadata_chain = metadata_prompt | llm | StrOutputParser()
chain = {"doc_type": doc_type_chain} | RunnablePassthrough.assign(
    metadata=metadata_chain
)
res = chain.invoke({"img": image})
print(res)
