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

# Load values from a local .env file, such as API credentials and a custom API URL.
# Do this before creating the client so the client can read those settings.
load_dotenv()
    
# Create a client that connects to the local Ollama server.
# Ollama does not use this API key, but the OpenAI client requires a value.
client = OpenAI()

# Ask the selected model a question and wait until it returns a response.
response = client.responses.create(
    # Local Ollama model does does not support file inputs.
    model="gpt-5.6-luna",
    # Give the model its behavior instructions and the user's question.
    instructions=INSTRUCTIONS,
    tools=[{"type":"web_search"}],
    # A response can contain multiple messages. Here, we send one user message.
    input="Are there any known outages with Teams right now?"
)

# Display only the text answer, not the full response object.
print("--------WEB SEARCH RESPONSE--------")
print(response.output_text)