# !pip install transformers datasets torch pinecone-client langchain-community faiss-cpu sentence-transformers
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
#PROMPT1: QA
template = """Use the following context to answer the question at the end. 
If you don't know the answer, please think rationally and answer from your own knowledge base.
Context: {context}

Question: {question}
Answer: 
"""
QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

#PROMPT2: Normal prompting
template= """
        Please answer the question.
        Answer professionally, and where appropriate, in a Computer Science educational context.
        Question: {question}
        Response:
        """
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

def chat_interface(textbox, chat):
    subject = get_job(textbox) #find keywords to search jobs in API
    if subject != "":
        create_jobs_csv(subject, "london")

        loader = CSVLoader(file_path="job_listings.csv")
        documents = loader.load() # load data for retrieval

        d = text_split.split_documents(documents)
        db = FAISS.from_documents(d, embeddings)

        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
        qa = RetrievalQA.from_chain_type(llm=llm, 
            retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5}),  #ANY PARAMETERS? e.g. search_kwargs={k: 3}
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs, verbose=True)
        
        # agent = create_csv_agent(
        #     llm,
        #     "job_listings.csv",
        #     verbose=True,
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # )
        # res = agent.invoke(input_dict)
        input_dict = {'query': textbox}
        result = qa.invoke(input_dict)
        documents = result.get("source_documents", [])
        for i in documents:
            print (i)
        text = result['result']
        return text
    else: # If no jobs are found, normal prompting and response is done
        print("TAKING ELSE ROUTE")
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        input_dict = {'question': textbox}
        response_dict = llm_chain.invoke(input_dict)
        response = response_dict['text'].split("Response:")[1].strip()
        return response
    return subject

def main():
    # search_url = f'{BASE_URL}?keywords="itconsultant"&locationName='
    # search_response = get(search_url, auth=(reed_key, '')) # authentication header as the username, with the password left empty
    # if search_response.status_code == 200:
    #     job_listings = search_response.json()
    #     print(job_listings)
    # else:
    #     ("ERROR", search_response.status_code)
    # create_jobs_csv("consultant", "london")
    database_test = pd.DataFrame({
        "question": [
            "What are qualifications for Data Scientist?",
            "Can you recommend me a graduate role for Software Engineering?",
            "What are the job responsibility of a Computer Vision Engineer?",
            "Graduate role...",
            "Intern...",
            "Can you name and describe the practical skills a full stack engineer should have?"
        ],
        "ground_truth": [
            "The qualifications for Data Scientist include a background in Computer Science, Engineering, or a related field, proficiency in Python programming, machine learning libraries, commercial experience with Azure, R, SQL, PowerBI, and Tableau, machine learning, assimilation, and optimizing data visualization dashboards.",
            "",
            "A Computer Vision Engineer is responsible for designing, developing, and optimizing computer vision algorithms for SLAM applications in construction environments. They implement and integrate SLAM solutions into their autonomous machinery and on-site monitoring systems. They collaborate with cross-functional teams to understand project requirements and define technical specifications. They conduct research and experimentation to evaluate and adopt the latest advancements in computer vision technology. They perform rigorous testing and validation to ensure the reliability and performance of developed solutions in real-world construction.",
            "",
            "",
            "As a full stack engineer, it is important to have a solid understanding of the entire development process, from front-end to back-end. You should be proficient in HTML, CSS, and JavaScript for web development, as well as knowledge of database management and software design. Additionally, you should be familiar with the latest trends and technologies, such as React, Angular, and Vue.js for front-end development, and Node.js and Python for back-end development.",
            
        ]
    })
    data_cs_industry = pd.DataFrame({
        "question": [
            "What are the latest trends in artificial intelligence and machine learning?",
            "What are some emerging programming languages that are gaining popularity in the industry?",
            "I am a beginner that wants to get into Data Science, where should I start?",
            "I am a final-year Computer Science student wanting to find a graduate role in Cybersecurity. What are the practical skills required for a career in Cybersecurity that are currently in-demand?",
            "What are the essential skills required for a career in cybersecurity?",
            "What are some in-demand technical skills for aspiring data analysts?",
            "What are the career prospects for individuals with expertise in cybersecurity risk management?",
        ],
        "ground_truth": [
            "In the realm of artificial intelligence (AI) and machine learning (ML), several notable trends have emerged recently. Firstly, there's a growing focus on explainable AI (XAI), which aims to make AI models more transparent and understandable to humans, crucial for applications in fields like healthcare and finance where interpretability is paramount. Secondly, federated learning has gained traction, enabling training of ML models across decentralized devices while preserving data privacy, pivotal for IoT and edge computing scenarios. Additionally, reinforcement learning (RL) advancements, particularly in deep RL, have seen remarkable progress, empowering AI systems to make sequential decisions in dynamic environments, with applications spanning robotics, autonomous vehicles, and gaming. Lastly, the integration of AI with other technologies like blockchain for enhanced security and trustworthiness and with quantum computing for tackling complex optimization problems signifies promising directions for future research and innovation in the AI landscape.",
            "Several emerging programming languages are gaining traction in the industry due to their unique features and capabilities. One such language is Rust, known for its emphasis on safety, concurrency, and performance, making it suitable for systems programming where reliability and efficiency are critical. Another language on the rise is Julia, which specializes in numerical and scientific computing, offering high performance comparable to traditional languages like C and Fortran while maintaining a user-friendly syntax and extensive library support. Additionally, Kotlin, a statically typed language interoperable with Java, has become increasingly popular for Android app development, offering modern features and improved developer productivity. Lastly, Swift, developed by Apple, has gained momentum for iOS and macOS development, providing a concise and expressive syntax along with powerful features like optionals and automatic memory management. These emerging languages cater to specific niches and address evolving industry needs, showcasing their growing relevance and adoption in the programming landscape.",
            "Here is a few things to learn about Data Science to get you started: \nLearn Python or R: Choose one as your primary programming language. \nBasic Statistics: Understand mean, median, mode, standard deviation, and probability.\nData Manipulation: Learn Pandas (Python) or dplyr (R) for data cleaning and manipulation.\nData Visualization: Use Matplotlib, Seaborn (Python), or ggplot2 (R) for visualization.\nMachine Learning Basics: Start with linear regression, logistic regression, decision trees, and evaluation metrics.\nPractice: Work on projects using real-world datasets from sources like Kaggle.\nStay Updated: Follow online resources and communities for the latest trends and techniques.",
            "As a final-year Computer Science student aiming for a graduate role in cybersecurity, it's essential to focus on developing practical skills that are currently in high demand in the industry.\nSome of these key skills include:\n\n1. Knowledge of Networking: Understanding networking fundamentals, protocols (such as TCP/IP), and network architecture is crucial for identifying and mitigating security threats. Familiarize yourself with concepts like firewalls, routers, VPNs, and intrusion detection systems (IDS).\n\n2. Proficiency in Operating Systems: Gain proficiency in operating systems such as Linux and Windows, including command-line operations, system administration tasks, and security configurations. Being able to secure and harden operating systems is essential for protecting against common cybersecurity threats.\n\n3. Understanding of Cryptography: Cryptography is at the heart of cybersecurity, so having a solid understanding of encryption algorithms, cryptographic protocols, and cryptographic techniques is vital. Learn about symmetric and asymmetric encryption, digital signatures, hashing algorithms, and their applications in securing data and communications.\n\n4. Penetration Testing and Ethical Hacking: Develop skills in penetration testing and ethical hacking to identify vulnerabilities and assess the security posture of systems and networks. Familiarize yourself with tools and techniques used by ethical hackers, such as Kali Linux, Metasploit, Nmap, and Wireshark.\n\n5. Security Assessment and Risk Management: Learn how to conduct security assessments, risk assessments, and threat modeling to identify, prioritize, and mitigate security risks effectively. Understand risk management frameworks like NIST, ISO 27001, and COBIT, and how to apply them in real-world scenarios.\n\n6. Incident Response and Forensics: Acquire knowledge of incident response procedures, including detection, analysis, containment, eradication, and recovery from security incidents. Understand digital forensics principles and techniques for investigating and analyzing security breaches and cybercrimes.\n\n7. Security Awareness and Communication: Develop strong communication skills to effectively convey cybersecurity concepts, risks, and recommendations to technical and non-technical stakeholders. Being able to raise awareness about cybersecurity best practices and policies is essential for promoting a security-conscious culture within organizations.\n\n8. Continuous Learning and Adaptability: Cybersecurity is a rapidly evolving field, so it's essential to cultivate a mindset of continuous learning and adaptability. Stay updated with the latest threats, trends, technologies, and best practices through professional development, certifications, and participation in cybersecurity communities and events.\n\nBy focusing on developing these practical skills and staying abreast of industry trends and advancements, you'll be well-prepared to pursue a successful career in cybersecurity upon graduation. Additionally, consider obtaining relevant certifications such as CompTIA Security+, CEH (Certified Ethical Hacker), CISSP (Certified Information Systems Security Professional), or others to further enhance your credentials and marketability in the field.",
            "A career in cybersecurity require a broad spectrum of practical skills, including proficiency in network security protocols and tools like firewalls and intrusion detection/prevention systems (IDS/IPS) for safeguarding network infrastructure. Secure coding practices and knowledge of common vulnerabilities are essential for developing secure software applications, with expertise in frameworks like OWASP Top 10 aiding in vulnerability mitigation. Encryption techniques and cryptographic protocols are vital for securing sensitive data, while incident response and digital forensics skills, alongside tools like SIEM systems, enable effective threat detection and response. Proficiency in penetration testing frameworks like Metasploit and security assessment tools is crucial for identifying and remediating security weaknesses, while knowledge of compliance frameworks such as GDPR ensures organizational adherence to cybersecurity regulations. Effective communication and collaboration skills are imperative for conveying cybersecurity risks and recommendations to stakeholders and collaborating with cross-functional teams to implement security measures. Continued learning and staying updated with the latest cybersecurity trends and technologies are key for navigating this ever-evolving field successfully.",
            "In-demand technical skills for data analysts include proficiency in programming languages like Python, R, or SQL for data manipulation, analysis, and visualization. Familiarity with statistical analysis techniques, such as regression analysis, hypothesis testing, and predictive modeling, is essential for deriving insights from data. Knowledge of data querying and database management systems like MySQL, PostgreSQL, or MongoDB is valuable for accessing and organizing large datasets. Expertise in data wrangling techniques, using tools like pandas, dplyr, or data.table, enables cleaning and transforming raw data into actionable insights. Proficiency in data visualization libraries like Matplotlib, ggplot2, or seaborn is crucial for creating informative and visually appealing charts, graphs, and dashboards to communicate findings effectively. Additionally, experience with machine learning frameworks like scikit-learn or TensorFlow, along with knowledge of data mining techniques, enhances the ability to build predictive models and extract patterns from data.",
            "Career opportunities for individuals in cybersecurity risk management include roles such as cybersecurity risk analysts, security consultants, risk managers, compliance officers, and cybersecurity architects. These professionals play a critical role in identifying, evaluating, and prioritizing cybersecurity risks, developing risk mitigation strategies, and ensuring compliance with regulatory requirements and industry standards. With the ever-evolving threat landscape and the increasing complexity of cybersecurity challenges, individuals with expertise in cybersecurity risk management can expect to have a wide range of career opportunities and advancement prospects in both the public and private sectors, including government agencies, financial institutions, healthcare organizations, and consulting firms. Additionally, obtaining relevant certifications such as Certified Information Systems Security Professional (CISSP), Certified Information Security Manager (CISM), or Certified Risk and Information Systems Control (CRISC) can further enhance career prospects and credibility in the field.",
        ]
    })

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