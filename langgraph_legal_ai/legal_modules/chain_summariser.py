#
# Chain Summariser
#
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from legal_modules.prompts import chain_summarizer_prompt
from legal_modules.setup import llm

chain_prompt_template = PromptTemplate(
    input_variables=["user_query", "legal_analysis"],
    template=chain_summarizer_prompt,
)

chain = chain_prompt_template | llm | StrOutputParser()
