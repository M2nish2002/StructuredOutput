from langchain_ollama import ChatOllama
from typing import TypedDict


model=ChatOllama(model="mistral:7b")

class output(TypedDict):
    summary:str
    sentiment:str


structured_model=model.with_structured_output(output)

result=structured_model.invoke("people are just snowflakes they want (positive) (acting like clowns) (and treating a fictional character like a real person) you know what the fandom said when tighnari va been exposed to grooming children they said (aw i feel bad for tighnari he didn't deserve this he should have had a better va)")
print(result)