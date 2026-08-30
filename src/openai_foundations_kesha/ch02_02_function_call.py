# Import the OpenAI client and Python's JSON utilities.
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load settings such as OPENAI_API_KEY from the local .env file.
load_dotenv()

# Create the client used to send requests to the model.
client = OpenAI()

# These instructions tell the model how it should behave when answering.
INSTRUCTIONS = """
    You are an internal knowledge assistant for a technology company. 
    Answer employee questions about the company tools, policies, and technical topics. 
    Keep answers concise: 2-3 sentences max. If you don't know the answer, say so and suggest who to contact.
"""

# This dictionary acts as a small local employee database.
EMPLOYEES = {
"E001": {"name": "Maria Santos", "department": "Engineering", "role": "Senior Developer", "locations":"Austin"},
"E002": {"name": "James Chen", "department": "Marketing", "role": "Content Manager", "locations":"New York"},
"E003": {"name": "Priya Patel", "department": "IT Support", "role": "Help Desk Lead", "locations":"Atlanta"},
}

# This is the Python function the model can ask our program to run.
def lookup_employee(employee_id):
    # dict.get returns None instead of raising an error for an unknown ID.
    employee = EMPLOYEES.get(employee_id)
    if employee:
        # Tool outputs are sent back to the model as JSON strings.
        return json.dumps(employee)
    return json.dumps({"error": f"Employee ID {employee_id} not found."})

# Describe the function to the model using a JSON Schema.
# This does not run the function; it only tells the model when and how to call it.
tools = [
    {
        "type": "function",
        # The name must match the Python function handled below.
        "name": "lookup_employee",
        "description": "Look up an employee's details by their employee ID. Returns name, department, role, and location.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description":"The employee ID. e.g., E001" 
                }
            },
            "required": ["employee_id"],
            # Reject arguments that are not defined in properties.
            "additionalProperties": False,
        },
    }
]

print("--------MODEL REQUESTES A FUNCTION CALL--------")

# First request: give the model access to the tool and ask an employee question.
response = client.responses.create(
    model="gpt-5.6-luna",
    instructions=INSTRUCTIONS,
    tools=tools,
    input="Can you look up the details for employee ID E002?",
)

# A response can contain several output items, so inspect each one.
for item in response.output:
    if item.type == "function_call":
        print(f"Function call requested: {item.name}")
        print(f"Arguments: {item.arguments}")
        print()

        # Function arguments arrive as JSON text; parse them into a dictionary.
        arguments = json.loads(item.arguments)
        # Our application, not the model, executes the real Python function.
        result = lookup_employee(arguments["employee_id"])

        print("--------FUNCTION CALL RESPONSE--------")
        print(f"Function call result: {result}")
        print()

# Second request: return the tool result so the model can write a final answer.
final_response = client.responses.create(
    model="gpt-5.6-luna",
    instructions=INSTRUCTIONS,
    # Continue the same conversation rather than starting a new one.
    previous_response_id=response.id,
    input=[
        {
            "type": "function_call_output",
            # Match this result to the exact function call the model requested.
            "call_id": item.call_id,
            "output": result
        }
    ]
)

print("--------FINAL RESPONSE FROM MODEL--------")
print(final_response.output_text)