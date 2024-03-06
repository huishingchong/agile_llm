# !pip install transformers datasets torch pinecone-client langchain-community faiss-cpu sentence-transformers
from getpass import getpass
from dotenv import load_dotenv
import os
from pathlib import Path
import gradio as gr

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceEndpoint


env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if not huggingface_api_token:
    huggingface_api_token = getpass("Enter your Hugging Face Hub API token: ")

# specify HuggingFace model
model_name = "tiiuae/falcon-7b-instruct"
# llm = HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature":0.5, "max_length":1024, "max_new_tokens":200})

llm = HuggingFaceEndpoint(
    repo_id=model_name,
    model=model_name,
    task="text-generation",
    temperature=0.5,
    # max_length:1024,
    max_new_tokens=200
)
# prompting with langchain

# template = """
# You are a content recommender for a Computer Science Online Learning platform. 
# Your task is to suggest educational content for a Computer Science learner and answer the prompt. Given the user
# input prompt, generate a reply with generated topic of educational content for learner
# to learn to help fulfil their goal. Then suggest skills to learn based on current workforce demands.
# Make sure to use specific Computer Science terms and correct, timely, factual information in the response.
# Question: {question}
# Response: 
# """

template = """Use the following context to answer the question at the end. 
If you don't know the answer, please think rationally and answer from your own knowledge base.
Context: {context}

Question: {question}
Answer: 
"""

QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
# llm_chain = load_qa_chain(llm, chain_type="stuff")


# API

# RAG from synthetic data set
loader = CSVLoader(file_path="skills_build.csv")
documents = loader.load() # load data for retrieval


modelPath = "sentence-transformers/gtr-t5-base" # Using t5 sentence transformer model to generate embeddings
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': True} # Normalizing embeddings may help improve similarity metrics by ensuring that embeddings magnitude does not affect the similarity scores

# Initialise an instance of HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
d = text_split.split_documents(documents)
    # https://python.langchain.com/docs/integrations/vectorstores/faiss
db = FAISS.from_documents(d, embeddings)

chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm, 
                                 retriever=db.as_retriever(), 
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs, verbose=True)

def chat_interface(textbox, chat):
    input_dict = {'query': textbox}
    result = qa.invoke(input_dict)
    print(result)
    text = result['result']
    return text

def main():
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
    # Prepare dataset containing input output example types
    # Load dataset from HuggingFace
# dataset = load_dataset("Locutusque/UltraTextbooks")

# data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
# data = load_dataset('fever', 'wiki_pages')
# data["train"][100]
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_datasets = data.map(tokenize_function, batched=True)
# print(tokenized_datasets)


# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=api_key)
# pc.create_index(
#     name="quickstart",
#     dimension=8,
#     metric="euclidean",
#     spec=ServerlessSpec(
#         cloud='aws', 
#         region='us-west-2'
#     )
# )


# RAG from Wikipedia


# Evaluation

if __name__ == "__main__":
    main()