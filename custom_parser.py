from langchain.chains.llm import LLMChain
from langchain.output_parsers import OutputFixingParser, CommaSeparatedListOutputParser
from langchain.output_parsers.prompts import NAIVE_FIX_PROMPT

class CustomOutputFixingParser(OutputFixingParser):
    def __init__(self, llm):
        parser = CommaSeparatedListOutputParser()
        llm_chain = LLMChain(llm=llm, prompt=NAIVE_FIX_PROMPT)
        super().__init__(parser=parser, retry_chain=llm_chain)
