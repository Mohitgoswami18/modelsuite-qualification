from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

class Feedback(BaseModel): 
    sentiment: Literal['pos', 'neg'] = Field(description="sentiment of the text")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

model = ChatHuggingFace(llm=llm)


template1 = PromptTemplate(template="determine the sentiment of the feedback \n {topic} {format_intructions}", input_variables=['topic'], partial_variables={'format_intructions': parser2.get_format_instructions()})

prompt1 = template1.invoke("this is the worst mobile phone")

result = prompt1 | model | parser1 
print(result)

# template2 = PromptTemplate(template="generate mcq questions from the given topic \n {topic}", input_variables=['topic'])