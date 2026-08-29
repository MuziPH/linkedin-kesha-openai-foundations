# Import the OpenAI-compatible client used to communicate with Ollama.
from openai import OpenAI
# Import Python's built-in tool for converting binary image data to text.
import base64

# These instructions tell the model how it should behave when answering.
INSTRUCTIONS = """
    You are an internal knowledge assistant for a technology company. 
    Answer employee questions about the company tools, policies, and technical topics. 
    Keep answers concise: 2-3 sentences max. If you don't know the answer, say so and suggest who to contact.
"""

# Create a client that connects to the local Ollama server.
# Ollama does not use this API key, but the OpenAI client requires a value.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="YOUR_API_KEY")

# Open the image in binary mode ("rb") because images are not plain text.
with open("../../data/error.png", "rb") as error_image:
    # Read the image and convert it to base64 text so it can be sent in an API request.
    base64_encoded_image = base64.b64encode(error_image.read()).decode("utf-8")

#
# Ask the selected model a question and wait until it returns a response.
response = client.responses.create(
    # This is the name of the Ollama model to use.
    model="llava:7b",
    # Give the model its behavior instructions and the user's question.
    instructions=INSTRUCTIONS,
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "I'm getting this error. What does it mean and how do I fix it?"
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64_encoded_image}"
                }
            ]
        }
    ]
)

# Display only the text answer, not the full response object.
print(response.output_text)