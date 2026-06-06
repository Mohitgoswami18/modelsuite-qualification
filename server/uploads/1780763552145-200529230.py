arguments = response_with_tools.additional_kwargs['tool_calls'][0]['function']

# # Tool calling
# import json
# print(arguments)
# tool_response = multiply.invoke(arguments)

# message.append(tool_response)

# final_output = llm_with_tool.invoke(message)
# print(final_output.content)