from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.messages import HumanMessage, AIMessage
from dotenv import load_dotenv 

load_dotenv() 

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.5-35B-A3B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# result = model.invoke("hello")
# print(result.content)

#  Tool Binding 

#  Creation of a tool 
from langchain_core.tools import tool 

@tool
def multiply(a: int, b: int) -> int: 
    """ Multiplies two integers and returns the result."""

    return a*b 

# print(type(multiply))
# result = multiply.invoke({'a': 3, 'b': 4})
# print(result)
# print(multiply.name)
# print(multiply.args)
# print(multiply.description)

llm_with_tool = model.bind_tools([multiply])

message = [HumanMessage(content="what is the multiplication of 3 and 4")] 
print(message)
# response_without_tool = model.invoke(message)

response_with_tools = llm_with_tool.invoke(message)
message.append(response_with_tools)

arguments = response_with_tools.additional_kwargs['tool_calls'][0]['function']

# Tool calling
import json
print(arguments)
tool_response = multiply.invoke(arguments)
message.append(tool_response)

final_output = llm_with_tool.invoke(message)
print(final_output.content)