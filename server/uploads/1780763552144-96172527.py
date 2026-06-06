from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(template="give me a detailed summary on the topic \n {topic}", input_variables=['topic'])

template2 = PromptTemplate(template="give 2 pointer summary from the given detailed summary \n {summary}", input_variables=['summary'])

# Linear Chaining
chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic': 'langchain'})

print(result)
