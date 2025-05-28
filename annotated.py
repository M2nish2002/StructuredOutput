from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated


model=ChatOllama(model="mistral:7b")

class output(TypedDict):
    summary:Annotated[str,"return summary pof the provided passage"]
    sentiment:Annotated[str,"return Positive ,negative or neutral"]


structured_model=model.with_structured_output(output)

result=structured_model.invoke("After months of anticipation, I finally embarked on my dream vacation. At first, everything seemed perfect—the sun was shining, the ocean sparkled, and the hotel lobby looked impressive. But soon, things took a turn. My room was unclean, the air conditioning didn’t work, and the staff seemed indifferent to complaints. The nearby beach was closed for maintenance, and it rained during most of my stay. Though there were a few nice moments, like a delicious local meal and a friendly tour guide, the overall experience left me feeling let down and frustrated. It wasn’t the getaway I had hoped for")
print(result)