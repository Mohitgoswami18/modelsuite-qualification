from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task= "text-generation"
)

model = ChatHuggingFace(llm = llm)
history = []

while(True): 
    prompt = input("You: ")
    if prompt == 'exit': 
        break

    history.append("user: "+prompt)
    result = model.invoke(history)
    history.append("model: "+result.content)
    print("Model: ",result.content)

print(history)
