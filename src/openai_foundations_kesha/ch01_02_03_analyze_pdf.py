import os
# Import the OpenAI-compatible client used to communicate with Ollama.
from openai import OpenAI
# Import Python's built-in tool for converting binary image data to text.
import base64
from dotenv import load_dotenv

# These instructions tell the model how it should behave when answering.
INSTRUCTIONS = """
    You are an internal knowledge assistant for a technology company. 
    Answer employee questions about the company tools, policies, and technical topics. 
    Keep answers concise: 2-3 sentences max. If you don't know the answer, say so and suggest who to contact.
"""

load_dotenv()  # Load environment variables from .env file
    
# Create a client that connects to the local Ollama server.
# Ollama does not use this API key, but the OpenAI client requires a value.
client = OpenAI()

# Open the PDF in binary mode ("rb") because PDFs are not plain text.
with open("../../data/company_vpn_policy.pdf", "rb") as pdf_file:
    # Read the PDF and convert it to base64 text so it can be sent in an API request.
    base64_encoded_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

# Ask the selected model a question and wait until it returns a response.
response = client.responses.create(
    # Local Ollama model does does not support file inputs.
    model="gpt-5.6-luna",
    # Give the model its behavior instructions and the user's question.
    instructions=INSTRUCTIONS,
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "According to this policy, can I use my personal laptop to connect to the company VPN?"
                },
                {
                    "type": "input_file",
                    "filename": "company_vpn_policy.pdf",
                    "file_data": f"data:application/pdf;base64,{base64_encoded_pdf}"
                }
            ]
        }
    ]
)

# Display only the text answer, not the full response object.
print(response.output_text)