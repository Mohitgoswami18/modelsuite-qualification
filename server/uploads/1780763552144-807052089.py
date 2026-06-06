from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
from langchain_core import prompts
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    task="text-generation"
)

# prompts are used for dynamic prompting 

template = prompts.PromptTemplate(template="what is the capital of {city}?", input_variables=['city'])

prompt = template.invoke({'city':'india'})
print(prompt)

model = ChatHuggingFace(llm=llm)

result = model.invoke(prompt)

print(result.content)