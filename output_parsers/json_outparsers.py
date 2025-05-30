from Api import hf_api
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_api
)
model=ChatHuggingFace(llm=llm)



parser2=JsonOutputParser()
 
template=PromptTemplate(
    template="write the age ,city,name of a historical figure \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction":parser2.get_format_instructions()}

)



print(model.invoke(template.format()).content)
