# !pip install transformers datasets torch
from getpass import getpass
from dotenv import load_dotenv
import os
from pathlib import Path
import gradio as gr

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from langchain import HuggingFaceHub
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if not huggingface_api_token:
    huggingface_api_token = getpass("Enter your Hugging Face Hub API token: ")

# specify HuggingFace model
model_name = "google/flan-t5-xxl"
# model_name = "tiiuae/falcon-7b"
# model_name = "openai-community/gpt2"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
# model_name = "databricks/dolly-v2-3b"
llm = HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature":0, "max_length":400})

# prompting with langchain

# template = """Question: {question}
# Answer: """

instructions = """
    Recommend topic 1
    Put content of topic 2 here.
    
    Recommend topic 2
    Put content of topic 2 here...
    And continue for all the content you recommend for the given topic.
"""

example = """
    Question: What is Cybersecurity?
    Response: 
    Cybersecurity is the practice of securing information technology systems and networks from cyber attacks.
    To study Cybersecurity, you need to learn about different type of cyber attacks, for example, spoofing, phishing.
    The current in demand market skills for cybersecurity is using anti-malware systems and batch scripting.
"""

template = """

You are a content recommender for a Computer Science Online Learning platform. 
Your task is to suggest educational content for a Computer Science learner. Given the user
input prompt, generate a reply with generated topic of educational content for learner
to learn to help fulfil their goal. Then suggest skills to learn based on current workforce demands.
An example response is 
    Question: What is Cybersecurity?
    Response: 
    Cybersecurity is the practice of securing information technology systems and networks from cyber attacks.
    To study Cybersecurity, you need to learn about different type of cyber attacks, for example, spoofing, phishing.
    The current in demand market skills for cybersecurity is using anti-malware systems and batch scripting.
Follow the example response.
Make sure to use specific Computer Science terms and correct, timely, factual information in the response.
Question: {question}
Response:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def chat_interface(textbox, chat):
    input_dict = {'question': textbox}
    response = llm_chain.run(input_dict)
    return response

gr.ChatInterface(
    fn=chat_interface,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
    title="Chatbot",
    description="Ask Chatbot any question",
    theme="soft",
    examples=["What does AI stand for?", "What is Software Engineering?", "What is Cybersecurity?"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()

# Fine-tune LLM?

# Load dataset from HuggingFace

# RAG from Wikipedia

# RAG from synthetic data set

# Evaluation