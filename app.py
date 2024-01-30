# !pip install transformers datasets torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from getpass import getpass
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if not huggingface_api_token:
    huggingface_api_token = getpass("Enter your Hugging Face Hub API token: ")

# specify HuggingFace model
model_name = "google/flan-t5-xxl"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompting with langchain

# Fine-tune LLM?
# RAG from Wikipedia

# Evaluation