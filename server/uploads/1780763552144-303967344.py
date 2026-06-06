from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(template="generate notes for the given topic \n {topic}", input_variables=['topic'])

template2 = PromptTemplate(template="generate mcq questions from the given topic \n {topic}", input_variables=['topic'])

template3 = PromptTemplate(template="Merge the notes and mcq questions given as\n {notes} {mcq_questions}", input_variables=['notes', 'mcq_questions'])

#parallel chaining

chain1= template1 | model | parser
chain2 = template2 | model | parser

parallel_chain = RunnableParallel({
    "notes": chain1, 
    "mcq_questions": chain2
})

output_chain = template3 | model | parser 
final_chain = parallel_chain | output_chain

# print the chain graph for better understanding of the flow of data in the chain
final_chain.get_graph().print_ascii()

result = final_chain.invoke({'topic': 'langchain'})

print(result)
