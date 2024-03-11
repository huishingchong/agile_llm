from getpass import getpass
from dotenv import load_dotenv
import os
from pathlib import Path
import gradio as gr
from requests import get
import csv
import re
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

def create_jobs_csv(job_name, location, reed_key):
    """Function to create a CSV file with job listings requested from the API."""
    BASE_URL = 'https://www.reed.co.uk/api/1.0/search'
    # Construct the request URL
    job_name = cleanhtml(job_name)
    search_url = f'{BASE_URL}?keywords={job_name}&locationName={location}'
    # Send the request
    search_response = get(search_url, auth=(reed_key, '')) # authentication header as the username, with the password left empty
    
    # Check if the request was successful
    if search_response.status_code == 200:
        job_listings = search_response.json()
        
        # Create or overwrite the CSV file
        with open('job_listings.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Job Title', 'Job Description', 'Location', 'Part-time', 'Full-time', 'Graduate', 'Minimum Salary'] 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Iterate through job listings
            for job in job_listings["results"]:
                len(job_listings["results"])
                job_id = job["jobId"]
                details_url = f'https://www.reed.co.uk/api/1.0/jobs/{job_id}'
                detail_response = get(details_url, auth=(reed_key, ''))
                detail = detail_response.json()
                job_title = detail.get("jobTitle", "")
                job_description = cleanhtml(detail.get("jobDescription", ""))
                location = detail.get("locationName", "")
                graduate = detail.get("graduate", "")
                keywords = detail.get("keywords", "")
                part_time = detail.get("partTime", "")
                full_time = detail.get("fullTime", "")
                min_salary = detail.get("minimumSalary", "")
                # Write job details to CSV
                writer.writerow({'Job Title': job_title, 'Job Description': job_description, 'Location': location, "Part-time": part_time, "Full-time": full_time, 'Graduate': graduate, 'Minimum Salary': min_salary})
    else:
        print(f'Error: {search_response.status_code}')

def get_job(query):
    """Helper function that returns the CS subject of the sentence to feed into job search."""
    helper_template = """
    Sentence: {query}
    Output only the Computer Science job title of the sentence, give one or two words.
    For example, the output of "What can a cybersecurity consultant contribute to a company?" is "cybersecurity consultant".
    For example, the output of "What are skills required for a data science job?" is "data science".
    The output is:
    """
    prompt = PromptTemplate(template=helper_template, input_variables=["query"])
    model_name = "tiiuae/falcon-7b-instruct"

    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        model=model_name,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=200
    )
    helper_llm = LLMChain(llm=llm, prompt=prompt)
    print(helper_llm)
    response = helper_llm.invoke(input=query)
    text = response["text"]
    # if "\n" in text:
    #     text = text.split("\n")[1].strip()
    return text

def cleanhtml(raw_html):
    """Helper function to clean HTML tags from text."""
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

def main():
    # Get Hugging Face Hub API token
    huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not huggingface_api_token:
        huggingface_api_token = getpass("Enter your Hugging Face Hub API token: ")

    # Specify HuggingFace model
    # model_name = "tiiuae/falcon-7b-instruct"
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        model=model_name,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=200
    )

    # Define prompts
    template_qa = """Use the following context to answer the question at the end. 
    If you don't know the answer, please think rationally and answer from your own knowledge base.
    Context: {context}
    Question: {question}
    Answer: 
    """
    QA_CHAIN_PROMPT = PromptTemplate(template=template_qa, input_variables=["context", "question"])

    template_prompt = """
    Please answer the question.
    Answer professionally, and where appropriate, in a Computer Science educational context.
    Question: {question}
    Response:
    """
    prompt = PromptTemplate(template=template_prompt, input_variables=["question"])

    # Initialize HuggingFaceEmbeddings
    modelPath = "sentence-transformers/gtr-t5-base"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Initialize text splitter
    text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    # Get Reed API key
    reed_key = os.getenv('REED_API_KEY')

    # Define chat interface function
    def chat_interface(textbox, chat):
        subject = get_job(textbox) # Find keywords to search jobs in API
        print("subject is", subject)
        
        create_jobs_csv(subject, "london", reed_key)
        with open('job_listings.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            first_row = next(reader, None)
            if first_row:
                loader = CSVLoader(file_path="job_listings.csv")
                documents = loader.load() # Load data for retrieval

                d = text_split.split_documents(documents)
                db = FAISS.from_documents(d, embeddings)

                chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
                qa = RetrievalQA.from_chain_type(llm=llm, 
                    retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5}),
                    return_source_documents=True,
                    chain_type_kwargs=chain_type_kwargs, verbose=True)
                
                input_dict = {'query': textbox}
                result = qa.invoke(input_dict)
                documents = result.get("source_documents", [])
                for i in documents:
                    print(i)
                text = result['result']
                return text
            else: # If no jobs are found, normal prompting and response is done
                llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
                input_dict = {'question': textbox}
                response_dict = llm_chain.invoke(input_dict)
                response = response_dict['text']
                return response

    # Launch chat interface
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

if __name__ == "__main__":
    main()
