# agile_llm

## Project description

In collaboration with IBM, this dissertation investigates the possibility of creating more agile, timely relevant content using prompt engineering and fine-tuned model, Retrieval-Augmented Generation (RAG) in adapted pipeline.

The final product is extracting job information from a real-time Job API (Reed) and perform relevant analysis to provide a response that is more timely and domain-specific to the specific Computer Science job in interest from the user query. This project hopes to guide online learning platforms or students by providing insights on, for example the skills or responsibilities, for the current workplace in Computer Science.

## Organisation

### 2 Google Collab notebooks

The notebooks detailing my exploration and effectivness of incremenets of my pipeline is in:

- finetuned_model.ipynb: comparing pre-trained and fine-tuned model with the same prompt
- job_analysis.ipynb: using fine-tuned model that was better performing in last step, and performing RAG on job API
  To run the Google collab notebooks, Pro subscription would be required, because evaluation implemented require GPU use and additional memory. The specific runtime used was V100 GPU.

### Final implementation with Gradio interface

The final implementation/pipeline is in app.py which can be run locally on machine.
You must have an .env file in the same directory as app.py with your

- HUGGINGFACEHUB_API_TOKEN = ...
- REED_API_KEY = ...

In the terminal, run
`python3 app.py `
and a Gradio demo will open in a browser with specified web address.

### Testing

Methods functionality are tested in test_app.py. To run the test file, similarly to the prerequisites for app.py, you would need to have the .env file with your HuggingFace API and Reed API tokens.
