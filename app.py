# !pip install transformers datasets torch
from getpass import getpass
from dotenv import load_dotenv
import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain import HuggingFaceHub
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate 

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if not huggingface_api_token:
    huggingface_api_token = getpass("Enter your Hugging Face Hub API token: ")

# specify HuggingFace model
model_name = "google/flan-t5-small"
# model_name2 = "roberta-base"
llm = HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature":0, "max_length":64})

# prompting with langchain
template = """Question: {question}
 
Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Give me the definition of cybersecurity."
print(llm_chain.run(question))

# Fine-tune LLM?
# RAG from Wikipedia

# Evaluation