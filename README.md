# agile_llm

## Project description

In collaboration with IBM, this project investigates the possibility of creating more agile, timely relevant content using a combination of prompt engineering, a fine-tuned model, and Retrieval-Augmented Generation (RAG) in the adapted pipeline.

The final product is extracting job information from a real-time Job API (Reed) and perform relevant analysis to provide a response that is more timely and domain-specific to the specific Computer Science job in interest from the user query. This project hopes to guide online learning platforms or students by providing insights on, for example, the skills or responsibilities, of the current workplace in the tech industry.

## Organisation

### 2 Python notebooks

The Python notebooks detailing the exploration of LLM techniques and effectivness of incremenets of my pipeline are:

- finetuned_model.ipynb: comparing pre-trained and fine-tuned model with the same prompt
- job_analysis.ipynb: using fine-tuned model that was better performing in last step, and performing RAG on job API
  These notebooks are to be run on Google Colab and Pro subscription would be required, because evaluation implemented require GPU use and additional memory. Note that running the notebooks would use up compute units. The specific runtime used was V100 GPU.

### Final implementation with Gradio interface

The final implementation/pipeline is in app.py which can be run locally on machine. In order to run it, you must first install libraries and dependencies from requirements.txt with the command: `pip install -r requirements.txt`

You must have an .env file in the same directory as app.py with your

- HUGGINGFACEHUB_API_TOKEN = ...
- REED_API_KEY = ...

For guidance, the .env file should look like this:
![Alt text](env_file_image.png)
Get your HuggingFace token here: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token

You can sign up for the Reed API token here: https://www.reed.co.uk/developers/jobseeker

In the terminal, run
`python3 app.py `
and specified local web URL will be given where you can open the Gradio demo on your browser.

The interface will look like this:
![Alt text](interface_image.png)

Please be aware of the hourly usage limits of the Reed Job API.

Part of the pipeline is to automatically search for relevant jobs from the Reed Job API based on the user query, and the job listings returned is collected into the job_listings.csv file. Therefore the job_listings.csv file is to serve as the retrieval source for the LLM to perform Retrieval-Augmented Generation and is dynamically updated each time the user queries the chatbot.

### Testing

Methods functionality are tested in test_app.py. To run the test file, similarly to the prerequisites for app.py, you would need to have the .env file with your HuggingFace API and Reed API tokens.

### Evaluation CSV files

The 'cs' and 'industry' folders contain csv files containing output of the models that are generated from the notebooks for evaluation purposes. These can be viewed (for transparency), but can be ignored as they are included in the Appendix with additional manual evaluation annotations.
