from Api import hf_api
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_api
)
model=ChatHuggingFace(llm=llm)



template1=PromptTemplate(
    template="write a detailed report on {topic1}",
    input_variables=["topic1"]

)
template2=PromptTemplate(
    template="write a five line summaryon {topic2}",
    input_variables=["topic2"]

)

prompt1=template1.invoke({"topic1":"blackhole"})
result=model.invoke(prompt1)

prompt2=template2.invoke({"topic2":result.content})

print(model.invoke(prompt2).content)
