# !pip install transformers datasets torch pinecone-client langchain-community faiss-cpu sentence-transformers
from getpass import getpass
from dotenv import load_dotenv
import os
from pathlib import Path
import gradio as gr
from requests import get
import csv
import re

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
from langchain.agents import tool, AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if not huggingface_api_token:
    huggingface_api_token = getpass("Enter your Hugging Face Hub API token: ")

# specify HuggingFace model
model_name = "tiiuae/falcon-7b-instruct"
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

# Initialise an instance of HuggingFaceEmbeddings
modelPath = "sentence-transformers/gtr-t5-base" # Using t5 sentence transformer model to generate embeddings
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': True} # Normalizing embeddings may help improve similarity metrics by ensuring that embeddings magnitude does not affect the similarity scores
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# API
reed_key = os.getenv('REED_API_KEY')
BASE_URL = 'https://www.reed.co.uk/api/1.0/search'

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

def create_jobs_csv(job_name, location):
    # Construct the request URL
    job_name = cleanhtml(job_name)
    search_url = f'{BASE_URL}?keywords={job_name}&locationName={location}'
    print(search_url)
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

# agent_executor = AgentExecutor(agent=qa, tools=tools, verbose=True)


@tool
def get_job(query):
    """Returns the subject of the sentence, helper function to feed into job search."""
    helper_template = """
    Sentence: {query}
    Output only the subject of the sentence, give one or two words.
    """
    prompt = PromptTemplate(template=helper_template, input_variables=["query"])
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        model=model_name,
        task="text-generation",
        temperature=0.5,
        # max_length:1024,
        max_new_tokens=200
    )
    helper_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    print(helper_llm)
    response = helper_llm.invoke(input=query)
    text = response["text"]
    # if "\n" in text:
    #     text = text.split("\n")[1].strip()
    return text

keywords = {
    "software engineer": "Software Engineer",
    "web developer": "Web Developer",
    "data analyst": "Data Analyst",
    "cybersecurity": "Cybersecurity",
    "data science": "Data Science",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence"
}

# def identify_job_name(query: str, keywords: dict) -> str:
#     """Identify the job name from the query string based on the dictionary of keywords."""
#     job_name = ""
#     for keyword, job_name_candidate in keywords.items():
#         print(keyword)
#         if keyword.lower() in query.lower():
#             print("YES")
#             job_name = job_name_candidate
#             break  # Stop iterating once a match is found
#     print("THE JOB NAME IS", job_name)
#     return job_name


# def identify_job_name(query, keywords):
#     job_name = ""
#     for i in keywords:
#         if i.lower() in query.lower():
#             job_name = i
#     return job_name

def chat_interf(textbox, chat):
    subject = get_job(textbox)
    # subject = identify_job_name(textbox, keywords)
    if subject != "":
        create_jobs_csv(subject, "london")
        input_dict = {'query': textbox}

        loader = CSVLoader(file_path="job_listings.csv")
        documents = loader.load() # load data for retrieval

        d = text_split.split_documents(documents)
        db = FAISS.from_documents(d, embeddings)

        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
        qa = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=db.as_retriever(), 
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs, verbose=True)
        
        # agent = create_csv_agent(
        #     llm,
        #     "job_listings.csv",
        #     verbose=True,
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # )
        # res = agent.invoke(input_dict)
        result = qa.invoke(input_dict)
        print(result.get("source_documents", []))
        text = result['result']
        return text
    else:
        print("TAKING ELSE ROUTE")
        template= """
        Please answer the question.
        Answer professionally, and where appropriate, in a Computer Science educational context.
        Question: {question}
        Response:
        """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        input_dict = {'question': textbox}
        response_dict = llm_chain.invoke(input_dict)
        response = response_dict['text'].split("Response:")[1].strip()
        return response
    return subject


def chat_interface(textbox, chat):
    answer = get_job(textbox)
    print("the answer is:", answer)
    return answer

def main():
    # search_url = f'{BASE_URL}?keywords="itconsultant"&locationName='
    # search_response = get(search_url, auth=(reed_key, '')) # authentication header as the username, with the password left empty
    # if search_response.status_code == 200:
    #     job_listings = search_response.json()
    #     print(job_listings)
    # else:
    #     ("ERROR", search_response.status_code)
    # create_jobs_csv("consultant", "london")

    gr.ChatInterface(
        fn=chat_interf,
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
    
    # PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
    # FORMAT_INSTRUCTIONS = """Use the following format:
    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # Thought: I now know the final answer
    # Final Answer: the final answer to the original input question"""
    
    # SUFFIX = """Begin!
    # Question: {input}    """
    # agent = create_csv_agent(
    #     llm,
    #     "rag_sample.csv",
    #     verbose=True,
    #     agent_executor_kwargs={
    #         "handle_parsing_errors": True,
    #         'prefix':PREFIX,
    #         'format_instructions':FORMAT_INSTRUCTIONS,
    #         'suffix':SUFFIX
    #     }
        # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # )

    # textbox = "how many rows are there?"
    # input_dict = {'input': textbox}
    # # res = agent.invoke(input_dict)
    # res = agent.run(input_dict)
    # print(res)


if __name__ == "__main__":
    main()