from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
import pinecone
from dotenv import load_dotenv
import os
import tf_keras as keras

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initialize the Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

index = pc.Index(index_name)

#Creating embeddings for each of the text chunks
docsearch=PineconeStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)

