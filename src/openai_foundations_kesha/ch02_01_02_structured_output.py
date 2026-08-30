import os
# Import the OpenAI-compatible client used to communicate with Ollama.
from openai import OpenAI
# Import Python's built-in tool for converting binary image data to text.
import base64
# The model replies with JSON text, so we need json to turn it into a Python dict.
import json
from dotenv import load_dotenv

# These instructions tell the model how it should behave when answering.
INSTRUCTIONS = """
    You are an internal knowledge assistant for a technology company. 
    Answer employee questions about the company tools, policies, and technical topics. 
    Keep answers concise: 2-3 sentences max. If you don't know the answer, say so and suggest who to contact.
"""

# A JSON Schema describing the exact shape of the reply we want back.
# Without this, the model returns free-form text that is harder to work with.
SCHEMA = {
    # The reply must be a JSON object (like a Python dictionary).
    "type": "object",
    # "properties" lists each field the object can contain.
    "properties": {
        "answer": {
            "type": "string",
            # Descriptions are hints that help the model fill each field correctly.
            "description": "The answer to the employee's question.",
        },
        "sources": {
            # "array" means a list; "items" says every entry in it must be a string.
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of URLs or references used to answer the question.",
        },
        "confidence": {
            "type": "string",
            # "enum" restricts the value to one of these three choices.
            "enum": ["low", "medium", "high"],
            "description": "The model's confidence in the answer.",
        },
    },
    # Every field listed here must be present in the reply.
    "required": ["answer", "sources", "confidence"],
    # Block any extra fields we did not ask for.
    "additionalProperties": False,
}

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
    # Let the model look up live information on the web before answering.
    tools=[{"type": "web_search"}],
    # A response can contain multiple messages. Here, we send one user message.
    input="Are there any known outages with Teams right now?",
    # Ask for the reply as JSON that matches SCHEMA instead of plain text.
    text={
        "format": {
            "type": "json_schema",
            # A label for this format; any descriptive name works.
            "name": "structured_output",
            # "strict" forces the model to follow the schema exactly.
            "strict": True,
            "schema": SCHEMA,
        }
    }
)

# The reply is JSON text, so convert it into a Python dictionary we can index.
result = json.loads(response.output_text)
# Display only the text answer, not the full response object.
print("--------WEB SEARCH STRUCTURED RESPONSE--------")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Confidence: {result['confidence']}")