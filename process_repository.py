#!/usr/bin/env python3
"""
Repository Content Processing Script

This script fetches repository content from gitingest URLs, chunks the text,
generates embeddings using BAAI model, and stores everything in Milvus vectorstore
with extracted metadata.
"""

import os
import re
from bs4 import BeautifulSoup
import requests
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import uuid
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import argparse
import logging
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False
    logger.warning("SemanticChunker not available. Will use RecursiveCharacterTextSplitter only.")

BASE_URL = "https://gitingest.com"

class RepositoryProcessor:
    """Process repository content and store in Milvus with embeddings."""
    
    def __init__(self, collection_name: str = "repository_content", model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize the repository processor.
        
        Args:
            collection_name: Name of the Milvus collection
            model_name: BAAI embedding model name
        """
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Detect device (GPU if available, else CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding model with GPU support
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize LangChain embedding model for semantic chunking with GPU support
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Milvus connection
        self._connect_to_milvus()
        
        # Create or get collection
        self._setup_collection()
    
    def _connect_to_milvus(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host="localhost",
                port="19530"
            )
            logger.info("Successfully connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collection(self):
        """Create or get the Milvus collection with proper schema."""
        # Get the embedding dimension from the model
        sample_embedding = self.embedding_model.encode(["test"])
        embedding_dim = len(sample_embedding[0])
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="chunk_length", dtype=DataType.INT64),
            FieldSchema(name="relevant_urls", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="has_metadata", dtype=DataType.BOOL),
        ]
        
        schema = CollectionSchema(fields, f"Repository content collection")
        
        # Create collection if it doesn't exist
        if utility.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists")
            self.collection = Collection(self.collection_name)
        else:
            logger.info(f"Creating collection '{self.collection_name}'")
            self.collection = Collection(self.collection_name, schema)
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
            logger.info("Created vector index for collection")
        
        # Load collection
        self.collection.load()
    
    def fetch_gitingest_content(self, repo_url: str) -> str:
        """Scrape a GitHub repository using GitIngest with improved format detection"""
        try:
            if not repo_url.startswith("https://github.com/"):
                return f"Invalid GitHub URL: {repo_url}"
            
            repo_name = repo_url.replace("https://github.com/", "").strip("/")
            if not repo_name:
                return "Empty repository name"
            
            if not re.match(r"^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$", repo_name):
                return f"Invalid repository name format: {repo_name}"
            
            # Create a new session with better headers
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            })
            
            try:
                # Submit the form directly
                payload = {
                    "input_text": repo_name,
                    "pattern_type": "exclude",
                    "pattern": "",
                    "max_file_size": "512"
                }
                
                print(f"Submitting repository: {repo_name} to GitIngest")
                response = session.post(
                    BASE_URL,
                    data=payload,
                    headers={"Referer": BASE_URL},
                    timeout=30
                )
                response.raise_for_status()
                
                # Parse response for download link
                soup = BeautifulSoup(response.text, "html.parser")
                download_link_tag = soup.find("a", href=lambda href: href and "download" in href)
                
                if not download_link_tag:
                    return "Download link not found in GitIngest response"
                
                # Extract the download link
                download_link = download_link_tag["href"]
                if not download_link.startswith("http"):
                    download_link = BASE_URL + download_link
                
                print(f"Found download link: {download_link}")
                
                # Download the repository content
                response = session.get(download_link, timeout=30)
                response.raise_for_status()
                content = response.text
                
                # Try several patterns to identify files in the content
                file_contents = []
                
                # Pattern 1: Traditional separator pattern
                traditional_matches = re.finditer(r"^(={20,})\s*File:\s*(.*?)\s*\1", content, re.MULTILINE | re.DOTALL)
                
                # Pattern 2: Markdown style headers
                markdown_matches = re.finditer(r"^(#{1,6})\s+(.*?\.(?:py|js|java|cpp|h|c|go|rb|rs|php|html|css|tsx?|jsx?))\s*$", 
                                        content, re.MULTILINE)
                
                # Pattern 3: Filename followed by content
                filename_matches = re.finditer(r"^(.*?\.(?:py|js|java|cpp|h|c|go|rb|rs|php|html|css|tsx?|jsx?))\s*[:=]\s*$",
                                        content, re.MULTILINE)
                
                # Pattern 4: Special formatted filename blocks (like <file:path/to/file.py>)
                special_matches = re.finditer(r"[<\[](file|path):([^>\]]+\.[a-zA-Z]+)[>\]]", content)
                
                patterns_tried = []
                
                # Try traditional separator
                file_blocks = []
                for match in traditional_matches:
                    separator = match.group(1)
                    separator_pattern = rf"^{re.escape(separator)}\s*File:\s*(.*?)\s*{re.escape(separator)}"
                    file_blocks = list(re.finditer(separator_pattern, content, re.MULTILINE | re.DOTALL))
                    if file_blocks:
                        patterns_tried.append("traditional")
                        break
                
                if file_blocks:
                    for i, match in enumerate(file_blocks):
                        filename = match.group(1).strip()
                        content_start = match.end()
                        content_end = file_blocks[i+1].start() if i < len(file_blocks)-1 else len(content)
                        file_content = content[content_start:content_end].strip()
                        file_contents.append((filename, file_content))
                
                # If no files found, split the content by common patterns
                if not file_contents:
                    # Try a general approach by splitting on potential file markers
                    pattern = r"(^|\n)(?:---+\s*|\*{3,}\s*|#{3,}\s*|={3,}\s*)?(?:File|PATH|FILENAME)?\s*[:\-=]?\s*([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+)(?:\s*[:\-=]|\s*$)"
                    general_matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    
                    if general_matches:
                        patterns_tried.append("general")
                        for i, match in enumerate(general_matches):
                            filename = match.group(2).strip()
                            content_start = match.end()
                            content_end = general_matches[i+1].start() if i < len(general_matches)-1 else len(content)
                            file_content = content[content_start:content_end].strip()
                            file_contents.append((filename, file_content))
                
                # Last resort: try to find filenames and split content
                if not file_contents:
                    patterns_tried.append("filename_extraction")
                    filename_pattern = r"(?:^|\n)([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]{1,5})(?:\s*:|\s*$)"
                    filenames = list(re.finditer(filename_pattern, content))
                    
                    if filenames:
                        for i, match in enumerate(filenames):
                            filename = match.group(1).strip()
                            content_start = match.end()
                            content_end = filenames[i+1].start() if i < len(filenames)-1 else len(content)
                            file_content = content[content_start:content_end].strip()
                            if len(file_content) > 10:  # Ensure we have meaningful content
                                file_contents.append((filename, file_content))
                
                # If we still have no content, see if it looks like a single file
                if not file_contents and len(content) > 100:
                    patterns_tried.append("single_file")
                    # Extract likely filename from URL or repo name
                    parts = repo_name.split('/')
                    if len(parts) >= 2:
                        repo_dir = parts[-1]
                        likely_extension = ".py" if "python" in content.lower() or "def " in content else ".js"
                        filename = f"{repo_dir}/main{likely_extension}"
                        file_contents.append((filename, content))
                
                if not file_contents:
                    # Save content for debugging
                    with open("gitingest_content.txt", "w", encoding="utf-8") as f:
                        f.write(content[:5000])  # Save more content
                    return f"No file patterns identified. Tried: {', '.join(patterns_tried)}. See gitingest_content.txt"
                    
                # Format the output as before
                formatted_output = "\n".join([f"--- File: {fn} ---\n{cnt}" for fn, cnt in file_contents])
                with open("gitingest_output.txt", "w", encoding="utf-8") as f:
                    f.write(formatted_output)
                logger.info(f"Successfully scraped repository: {repo_name}")
                print(f"Successfully scraped repository: {repo_name}")
                return formatted_output
                
            except requests.RequestException as e:
                return f"Network error: {str(e)}"
                
        except Exception as e:
            import traceback
            print(f"Unexpected error in scrape_repo: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return f"Unexpected error: {str(e)}"
    
    def extract_metadata_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Extract metadata from text content, specifically looking for 'Page: url_to_information' patterns.
        
        Args:
            text: Raw text content
            
        Returns:
            List of dictionaries containing extracted metadata
        """
        metadata_entries = []
        
        # Pattern to match 'Page: url_to_information'
        page_pattern = r'Page:\s+(https?://[^\s\n]+)'
        
        matches = re.finditer(page_pattern, text, re.MULTILINE | re.IGNORECASE)
        
        for match in matches:
            url = match.group(1)
            # Get the position of the match to extract surrounding context
            start_pos = max(0, match.start() - 500)  # 500 chars before
            end_pos = min(len(text), match.end() + 1500)  # 1500 chars after
            
            context = text[start_pos:end_pos].strip()
            
            metadata_entries.append({
                'url': url,
                'context': context,
                'match_position': match.start()
            })
        
        logger.info(f"Extracted {len(metadata_entries)} metadata entries with URLs")
        return metadata_entries
    
    def chunk_text_semantic(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200, method: str = "semantic") -> List[str]:
        """
        Chunk text using LangChain splitters with semantic awareness.
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            method: Chunking method ('semantic' or 'recursive')
            
        Returns:
            List of text chunks
        """
        logger.info(f"Starting {method} text chunking with size {chunk_size}")
        
        if method == "semantic" and SEMANTIC_CHUNKER_AVAILABLE:
            try:
                # Use semantic chunker for better context-aware splitting
                text_splitter = SemanticChunker(
                    embeddings=self.langchain_embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=95
                )
                chunks = text_splitter.split_text(text)
                
                # If chunks are too large, use recursive splitter as fallback
                if any(len(chunk) > chunk_size * 2 for chunk in chunks):
                    logger.info("Semantic chunks too large, falling back to recursive splitter")
                    return self.chunk_text_semantic(text, chunk_size, chunk_overlap, "recursive")
                    
            except Exception as e:
                logger.warning(f"Semantic chunking failed: {e}. Falling back to recursive chunker.")
                return self.chunk_text_semantic(text, chunk_size, chunk_overlap, "recursive")
        else:
            if method == "semantic" and not SEMANTIC_CHUNKER_AVAILABLE:
                logger.warning("Semantic chunking requested but not available. Using recursive chunker.")
            # Fallback to recursive character text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=True,
                length_function=len,
            )
            chunks = text_splitter.split_text(text)
        
        # Filter out very small chunks
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        
        logger.info(f"Created {len(chunks)} text chunks using {method} method")
        return chunks
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in batches for better performance.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size} on {self.device}")
        
        all_embeddings = []
        
        # Adjust batch size based on device
        if self.device == 'cuda':
            # Use larger batch size for GPU
            batch_size = min(batch_size * 2, 64)
            logger.info(f"Using GPU - increased batch size to {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Generate embeddings for this batch with GPU acceleration
            with torch.no_grad():  # Reduce memory usage
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=batch_size,
                    device=self.device
                )
            all_embeddings.extend(batch_embeddings.tolist())
        
        return all_embeddings
    
    def store_in_milvus_batch(self, chunks: List[str], embeddings: List[List[float]], 
                               metadata_list: List[Dict], source_url: str, batch_size: int = 100):
        """
        Store chunks, embeddings, and metadata in Milvus in batches for better performance.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries
            source_url: Original gitingest URL
            batch_size: Number of items to store per batch
        """
        logger.info(f"Storing {len(chunks)} chunks in Milvus in batches of {batch_size}")
        
        # Prepare data for Milvus
        all_data = []
        for i, chunk in enumerate(chunks):
            # Find relevant URL metadata for this chunk
            relevant_urls = []
            for meta in metadata_list:
                # Check if this chunk contains or is near the metadata context
                if meta['context'] in chunk or any(word in chunk.lower() for word in meta['url'].split('/')[-2:]):
                    relevant_urls.append(meta['url'])
            
            data_entry = {
                'id': str(uuid.uuid4()),
                'document': chunk[:65535],  # Truncate if too long
                'embedding': embeddings[i],
                'source_url': source_url[:1000],  # Truncate if too long
                'chunk_index': i,
                'chunk_length': len(chunk),
                'relevant_urls': ','.join(relevant_urls)[:2000] if relevant_urls else '',  # Truncate if too long
                'has_metadata': len(relevant_urls) > 0
            }
            all_data.append(data_entry)
        
        # Insert data in batches
        for i in range(0, len(all_data), batch_size):
            batch_end = min(i + batch_size, len(all_data))
            batch_data = all_data[i:batch_end]
            
            logger.info(f"Storing batch {i//batch_size + 1}/{(len(all_data) + batch_size - 1)//batch_size}")
            
            # Prepare data for Milvus insert
            batch_insert_data = [
                [item['id'] for item in batch_data],           # ids
                [item['document'] for item in batch_data],     # documents
                [item['embedding'] for item in batch_data],    # embeddings
                [item['source_url'] for item in batch_data],   # source_urls
                [item['chunk_index'] for item in batch_data],  # chunk_indices
                [item['chunk_length'] for item in batch_data], # chunk_lengths
                [item['relevant_urls'] for item in batch_data], # relevant_urls
                [item['has_metadata'] for item in batch_data]  # has_metadata
            ]
            
            # Insert this batch into Milvus
            self.collection.insert(batch_insert_data)
            self.collection.flush()  # Ensure data is persisted
        
        logger.info(f"Successfully stored {len(chunks)} chunks in Milvus")
    
    def process_repository(self, gitingest_url: str, chunk_size: int = 1000, overlap: int = 200, chunking_method: str = "semantic"):
        """
        Complete pipeline to process repository from gitingest URL.
        
        Args:
            gitingest_url: URL to fetch repository content from
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            chunking_method: Method for chunking ('semantic' or 'recursive')
        """
        logger.info(f"Starting repository processing for: {gitingest_url}")
        
        # Step 1: Fetch content
        raw_content = self.fetch_gitingest_content(gitingest_url)
        
        # Step 2: Extract metadata
        metadata_entries = self.extract_metadata_from_text(raw_content)
        
        # Step 3: Chunk text using semantic splitting
        chunks = self.chunk_text_semantic(raw_content, chunk_size, overlap, chunking_method)
        
        # Step 4: Generate embeddings in batches
        embeddings = self.generate_embeddings_batch(chunks, batch_size=16)  # Smaller batch size for stability
        
        # Step 5: Store in Milvus in batches
        self.store_in_milvus_batch(chunks, embeddings, metadata_entries, gitingest_url, batch_size=50)
        
        logger.info("Repository processing completed successfully")
        
        # Print summary
        print(f"\n=== Processing Summary ===")
        print(f"Source URL: {gitingest_url}")
        print(f"Total content length: {len(raw_content):,} characters")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Metadata URLs found: {len(metadata_entries)}")
        print(f"Milvus collection size: {self.collection.num_entities}")
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict:
        """
        Search for similar content in the vectorstore.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results from Milvus
        """
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=n_results,
            output_fields=["document", "source_url", "relevant_urls", "has_metadata", "chunk_index"]
        )
        
        # Format results to match expected structure
        formatted_results = {
            "documents": [[hit.entity.get("document") for hit in results[0]]],
            "metadatas": [[{
                "source_url": hit.entity.get("source_url"),
                "relevant_urls": hit.entity.get("relevant_urls"),
                "has_metadata": hit.entity.get("has_metadata"),
                "chunk_index": hit.entity.get("chunk_index")
            } for hit in results[0]]],
            "distances": [[hit.distance for hit in results[0]]]
        }
        
        return formatted_results

def main():
    """Main function to run the repository processor."""
    parser = argparse.ArgumentParser(description='Process repository content with embeddings')
    parser.add_argument('gitingest_url', help='GitIngest URL to process')
    parser.add_argument('--collection-name', default='repository_content', help='Milvus collection name')
    parser.add_argument('--model', default='BAAI/bge-base-en-v1.5', help='Embedding model name')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Text chunk size')
    parser.add_argument('--overlap', type=int, default=200, help='Chunk overlap size')
    parser.add_argument('--chunking-method', default='semantic', choices=['semantic', 'recursive'], help='Chunking method')
    parser.add_argument('--search', help='Search query to test the vectorstore')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = RepositoryProcessor(args.collection_name, args.model)
        
        # Process repository
        processor.process_repository(
            args.gitingest_url,
            args.chunk_size,
            args.overlap,
            args.chunking_method
        )
        
        # Optional search test
        if args.search:
            print(f"\n=== Search Results for: '{args.search}' ===")
            results = processor.search_similar(args.search)
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                print(f"\nResult {i+1} (similarity: {1-distance:.3f}):")
                print(f"Source: {metadata.get('source_url', 'N/A')}")
                print(f"Relevant URLs: {metadata.get('relevant_urls', 'None')}")
                print(f"Content preview: {doc[:200]}...")
    
    except Exception as e:
        logger.error(f"Error processing repository: {e}")
        raise

if __name__ == "__main__":
    main()