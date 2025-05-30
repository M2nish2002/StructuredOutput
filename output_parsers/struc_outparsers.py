
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema


model=ChatOllama(model="mistral:7b")
 
schema=[
    ResponseSchema(name="fact1",description="fact 1 about the topic"),
    ResponseSchema(name="fact2",description="fact 2 about the topic"),
     ResponseSchema(name="fact3",description="fact 3 about the topic")
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="write 3 fact about the {name}\n {format_instruction}",
    input_variables=["name"],
    partial_variables={"format_instruction":parser.get_format_instructions()}

)
chain=template | model |parser
print(chain.invoke({"name":"blackhole"}))
