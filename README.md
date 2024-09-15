# Medecro GenAI Medical Chatbot

## Team name: The Qubits
### Team members:
- Akhilesh T S
- Karthik Sriram V

## Description
This Medical Chatbot is an advanced AI-powered application designed to assist users with medical inquiries. It offers three main functionalities:
1. Symptom Analysis
2. Medical Image Report Analysis
3. PDF Medical Report Analysis

The chatbot utilizes various AI models and technologies to provide informative responses and analyses.

## Features
- **Symptom Analysis**: Users can describe their symptoms, and the chatbot provides relevant medical information.
- **Image Report Analysis**: The chatbot can analyze uploaded medical images (in PNG format) and provide insights.
- **PDF Medical Report Analysis**: Users can upload medical reports in PDF format for the chatbot to analyze and explain in layman's terms.
- **Secure Authentication**: The application includes a basic authentication system to ensure secure access.

## Technologies Used
- Python
- Chainlit: For creating the interactive chat interface
- LangChain: For building the question-answering system
- Pinecone: For vector storage and similarity search
- HuggingFace: For embeddings
- OpenAI GPT-4: For advanced natural language processing tasks
- LLAMA: Local language model for certain tasks
- pdfplumber: For extracting text from PDF files

## Setup and Installation
1. Clone the repository:
   ```
   git clone https://github.com/akhilesh1709/Medecro-Gen-AI-Medical-Chatbot.git
   cd Medecro-Gen-AI-Medical-Chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Download the LLAMA model:
   Place the `llama-2-7b-chat.ggmlv3.q4_0.bin` file in the `model/` directory.

5. Set up Pinecone:
   Ensure you have a Pinecone index named "medical-chatbot" set up with the appropriate schema.

## Running the Application
To start the chatbot, run:
```
chainlit run app.py
```

## Usage
1. Access the chatbot through the provided URL after running the application.
2. Log in using the credentials (default: username: "admin", password: "admin").
3. Choose one of the three options:
   - Enter symptoms for analysis
   - Upload a medical image (PNG) for analysis
   - Upload a medical report (PDF) for analysis
4. Follow the prompts and receive AI-generated responses and analyses.

## Security Note
This application uses a basic authentication system. For production use, implement a more robust authentication and authorization system.

## Disclaimer
This chatbot is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Contributing
Contributions to improve the Medical Chatbot are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request
