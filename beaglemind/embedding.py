import time
from openai import AzureOpenAI
import os

from beaglemind.beagleenv import BeagleEnv

# Initialize as None to be instantiated later
openai_client = None

def connect_openai():
    global openai_client
    BeagleEnv.load_env_file()
    if openai_client is None:
        openai_client = AzureOpenAI(
            api_key=BeagleEnv.get_env("OPENAI_API_KEY"),
            api_version=BeagleEnv.get_env("OPENAI_API_VERSION"),
            azure_endpoint=BeagleEnv.get_env("OPENAI_AZURE_ENDPOINT"),
        )
    return openai_client

def get_openai_client():
    return connect_openai()

def get_embeddings(text: str):
    """
    Get embeddings for a given text using the OpenAI embedding model.
    :param text: The text to embed
    :return: The embeddings as a list of floats
    """
    client = get_openai_client()
    # Get embeddings
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding



class AzureOpenAIEmbeddings:
    """Wrapper for OpenAI embeddings to match LangChain interface"""
    
    def __init__(self):
        # Use the imported functions to access the client
        self.client = get_openai_client()
    
    def embed_documents(self, texts):
        """Create embeddings for a list of documents"""
        results = []
        # Process in smaller batches to avoid rate limits
        batch_size = 16  # Adjust as needed
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [get_embeddings(text) for text in batch]
            results.extend(batch_embeddings)
            if i + batch_size < len(texts):
                time.sleep(0.5)  # Avoid rate limiting
        return results
    
    def embed_query(self, text):
        """Create embedding for a single query"""
        return get_embeddings(text)