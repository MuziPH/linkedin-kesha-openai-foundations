# Import the OpenAI client, JSON utilities, environment loader, and OS helpers.
from openai import OpenAI
import json
from dotenv import load_dotenv
import os

# Load variables from the local .env file into the environment.
load_dotenv()

# Create the client used to send requests to the OpenAI Responses API.
client = OpenAI()

# Read the Context7 secret without placing it directly in the source code.
CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY")

# These instructions control how the model should answer every request below.
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

# This regular Python function can be called when the model requests employee data.
def lookup_employee(employee_id):
    employee = EMPLOYEES.get(employee_id)
    if employee:
        # Tool results are returned to the model as JSON text.
        return json.dumps(employee)
    return json.dumps({"error": f"Employee ID {employee_id} not found."})

# Describe the local function so the model knows its name and valid arguments.
employee_function_tool ={
    
        "type": "function",
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
            "additionalProperties": False,
        },
    }

# Describe the remote MCP server that the model may use as a documentation tool.
mcp_tool = {
    "type": "mcp",
    # The label identifies this server in model responses and logs.
    "server_label": "context7",
    "server_url": "https://mcp.context7.com/mcp",
    # Allow the model to use this server without asking for confirmation each time.
    "require_approval": "never",
    # Send the API key to Context7 when the OpenAI API connects to the server.
    "headers": {
        "Authorization": f"Bearer {CONTEXT7_API_KEY}"
    },
}


print("-------- MCP Documentation Search --------")

# Give the model access to Context7. The Responses API handles the remote MCP
# call and includes the server's information in the model's final answer.
response = client.responses.create(
    model="gpt-5.6-luna",
    instructions=INSTRUCTIONS,
    tools=[mcp_tool],
    input="How do I set up authentication in Next.js?",
)

print("--------MCP RESPONSE--------")
print(response.output_text)
print()


print("-------- Combined: Function + MCP Call --------")

# This request offers both a local function and a remote MCP server. The model
# can choose the appropriate tool for each part of the question.
comnined_response = client.responses.create(
    model="gpt-5.6-luna",
    instructions=INSTRUCTIONS,
    tools=[employee_function_tool, mcp_tool],
    input="What team dies employee E003 work on, and can you find the latest documentation on Python FastAPI?",
)

# Collect results for local function calls. MCP calls are handled automatically,
# but our application is responsible for executing regular Python functions.
function_call_outputs = []
for item in comnined_response.output:
    if item.type == "function_call":
        # Arguments arrive as JSON text, so convert them to a Python dictionary.
        arguments = json.loads(item.arguments)
        result = lookup_employee(arguments["employee_id"])
        # call_id connects this result to the model's original tool request.
        function_call_outputs.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": result
        })

if function_call_outputs:
    # Continue the same response and give the local results back to the model so
    # it can combine them with the MCP documentation into one natural answer.
    final_response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=INSTRUCTIONS,
        previous_response_id=comnined_response.id,
        input=function_call_outputs
    )

print("--------FINAL RESPONSE FROM MODEL--------")
print(final_response.output_text)