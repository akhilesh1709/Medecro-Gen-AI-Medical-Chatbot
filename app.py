import chainlit as cl
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import os
import base64
import pdfplumber
from openai import OpenAI

load_dotenv()

# Download and load the embeddings
embeddings = download_hugging_face_embeddings()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize the Pinecone index
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Now you can proceed to use the index
index = pc.Index(index_name)

# Loading the index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 512,
        'temperature': 0.8
    }
)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

# Initialize OpenAI client using environment variables
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Define prompts
sample_prompt = """You are a medical practitioner and an expert in analyzing medical-related images..."""

symptom_prompt = """You are a medical AI assistant. A user has described their symptoms..."""

# Prompt for analyzing medical reports
report_analysis_prompt = """You are an expert in understanding and interpreting medical reports. The user will provide a medical report, and you need to analyze it thoroughly. Identify key findings, possible diagnoses, and next steps for the patient. Please explain everything in simple terms that a layperson can understand. Make sure to include the disclaimer 'This analysis is not a substitute for professional medical advice. Please consult a doctor for an accurate diagnosis and treatment plan.'"""

# Function to encode image in base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to call GPT-4 model for image analysis
def call_gpt4_model_for_analysis(base64_image: str, sample_prompt=sample_prompt):
    messages = [
        {
            "role": "user",
            "content": sample_prompt
        },
        {
            "role": "user",
            "content": f"data:image/png;base64,{base64_image}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1500
    )

    return response.choices[0].message.content

# Function to call the QA chain for analyzing symptoms
def analyze_symptoms_with_qa(symptoms):
    # Use the QA chain to analyze the symptoms
    result = qa({"query": symptoms})
    return result["result"]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to analyze the medical report
def analyze_medical_report(pdf_text):
    messages = [
        {
            "role": "user",
            "content": f"{report_analysis_prompt}\n\nMedical Report Content:\n{pdf_text}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=1500
    )

    return response.choices[0].message.content

# Function to display the action buttons
async def display_action_buttons():
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(name="Ask symptoms", value="symptoms", label="Symptoms"),
            cl.Action(name="Analyze report", value="report", label="Image Report"),
            cl.Action(name="Analyze medical report", value="pdf_report", label="PDF Medical Report"),
        ],
    ).send()

    return res

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def main():
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello {app_user.identifier}").send()
    await cl.Message(content="Welcome to the Medical Chatbot! How can I assist you today?").send()

    # Display action buttons
    res = await display_action_buttons()

    while res:
        if res.get("value") == "symptoms":
            user_symptoms = await cl.AskUserMessage(content="Please enter your symptoms so that we can help you further.", timeout=5000).send()
            response = analyze_symptoms_with_qa(user_symptoms['output'])
            await cl.Message(content=str(response)).send()
            res = await display_action_buttons()

        elif res.get("value") == "report":
            await cl.Message(content="Please provide your medical image so that we can help you diagnose").send()  
            file = await cl.AskFileMessage(
                content="Please upload an image file (in .png format) to begin!",
                accept={"image/png": [".png"]}
            ).send()

            # Ensure the file is uploaded properly
            if file and isinstance(file, list) and len(file) > 0:
                uploaded_file = file[0]  # Access the first file in the list
                file_path = uploaded_file.path  # Get the file path
                base64_image = encode_image(file_path)
                response = call_gpt4_model_for_analysis(base64_image)  # Pass base64 image to GPT-4 analysis
                await cl.Message(content=str(response)).send()
            else:
                await cl.Message(content="No valid file was provided. Please try again.").send()
            # Show action buttons again after response
            res = await display_action_buttons()

        elif res.get("value") == "pdf_report":
            await cl.Message(content="Please upload your medical report (in PDF format) to analyze.").send()

            file = await cl.AskFileMessage(
                content="Please upload a PDF file of your medical report to begin!",
                accept={"application/pdf": [".pdf"]}
            ).send()

            # Ensure the PDF file is uploaded properly
            if file and isinstance(file, list) and len(file) > 0:
                uploaded_file = file[0]  # Access the first file in the list
                file_path = uploaded_file.path  # Get the file path

                # Extract text from the PDF
                pdf_text = extract_text_from_pdf(file_path)

                # Analyze the extracted text using GPT-4
                response = analyze_medical_report(pdf_text)
                await cl.Message(content=str(response)).send()
            else:
                await cl.Message(content="No valid PDF file was provided. Please try again.").send()
            # Show action buttons again after response
            res = await display_action_buttons()


if __name__ == "__main__":
    cl.run()