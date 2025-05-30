from Api import hf_api
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
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

parser=StrOutputParser()
 
chain=template1 | model | parser | model | parser

print(chain.invoke({"topic1":"blackhole"}))
