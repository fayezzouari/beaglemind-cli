import time
from openai import AzureOpenAI
import os

import requests
from langchain.embeddings.base import Embeddings
import time
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



class JinaAIEmbeddings(Embeddings):
    def __init__(self, api_key, model="jina-clip-v2", dimensions=1024):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = 'https://api.jina.ai/v1/embeddings'
        self.max_retries = 3
        self.retry_delay = 2
    
    def _get_embeddings(self, texts):
        """Helper function to get embeddings with retry logic and error handling"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            "model": self.model,
            "dimensions": self.dimensions,
            "normalized": True,
            "embedding_type": "float",
            "input": texts
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=data)
                
                # Debug the API response
                if response.status_code != 200:
                    print(f"API Error: Status code {response.status_code}")
                    print(f"Response: {response.text}")
                    if attempt < self.max_retries - 1:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise ValueError(f"API error after {self.max_retries} retries: {response.text}")
                
                # Try to parse the JSON response
                response_json = response.json()
                
                # Check if 'data' key exists in the response
                if 'data' not in response_json:
                    print(f"API Response format error: 'data' key missing")
                    print(f"Response: {response_json}")
                    
                    # If there's an error message, return it
                    if 'error' in response_json:
                        raise ValueError(f"API returned error: {response_json['error']}")
                    
                    # Handle potential different response formats
                    if 'embeddings' in response_json:
                        return response_json['embeddings']
                    
                    # If this is just a single embedding
                    if 'embedding' in response_json:
                        return [response_json['embedding']]
                    
                    # As a fallback, check if the response is directly a list of embeddings
                    if isinstance(response_json, list) and all(isinstance(item, list) for item in response_json):
                        return response_json
                    
                    raise ValueError(f"Unexpected API response format: {response_json}")
                
                # Normal case - extract embeddings from data field
                return [item['embedding'] for item in response_json['data']]
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"Request error: {e}. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise ValueError(f"Request failed after {self.max_retries} retries: {e}")
    
    def embed_documents(self, texts):
        """Embeds multiple texts at once (for Chroma compatibility)"""
        if not texts:
            return []
        
        # Split into smaller batches if needed (Jina API typically has limits)
        batch_size = 100  # Adjust based on API limits
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
            batch_embeddings = self._get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Sleep briefly between batches to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.5)
        print(all_embeddings)
        return all_embeddings
    
    def embed_query(self, text):
        """Embeds a single query text"""
        result = self.embed_documents([text])
        print("result", result)
        if result:
            return result[0]
        return []