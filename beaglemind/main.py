import datetime
import json
import os
import logging
import uuid
import click
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import uuid
import logging
import click
import re

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# Other necessary imports
from langchain_groq import ChatGroq
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from beaglemind.db_repair import VectorDBRepair
from beaglemind.embedding import AzureOpenAIEmbeddings, JinaAIEmbeddings
from beaglemind.beagleenv import BeagleEnv

# Disable PostHog tracking
os.environ['POSTHOG_DISABLE_SEND'] = 'true'

class ChatState(TypedDict):
    """Enhanced state for the chatbot with more precise tracking."""
    messages: list[AnyMessage]
    retrieved_context: list[str]
    chat_history: list[AnyMessage]

    

# Helper functions to serialize and deserialize messages
def serialize_message(message):
    if isinstance(message, HumanMessage):
        return {"role": "human", "content": message.content}
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "ai", "content": message.content}
    else:
        return {"role": "unknown", "content": str(message)}

def deserialize_message(message_dict):
    role = message_dict.get("role")
    content = message_dict.get("content")
    if role == "human":
        return HumanMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "ai":
        return AIMessage(content=content)
    else:
        return HumanMessage(content=content)

class BeagleMindAgent:
    def __init__(self, memory_file="conversation_memory.json"):
        """Initialize the BeagleMind agent with improved database handling."""
        load_dotenv()
        
        # API Key validation
        BeagleEnv.load_env_file()
        self.memory_file = Path(memory_file)
        self.collection_name = "repository_content"  # Store the collection name
        
        # Initialize conversation memory
        if self.memory_file.exists():
            try:
                raw = self.memory_file.read_text().strip()
                if not raw:
                    self.conversation_memory = {}
                else:
                    raw_memory = json.loads(raw)
                    self.conversation_memory = {
                        thread_id: [deserialize_message(msg) for msg in chat_history]
                        for thread_id, chat_history in raw_memory.items()
                    }
            except Exception as e:
                logging.warning("Conversation memory file is malformed. Starting with an empty memory.")
                self.conversation_memory = {}
        else:
            self.conversation_memory = {}

        # Initialize thread ID
        self.thread_id_file = Path("thread_id.txt")
        if self.thread_id_file.exists():
            self.thread_id = self.thread_id_file.read_text().strip()
        else:
            self.thread_id = f"session_{uuid.uuid4().hex[:8]}"
            self.thread_id_file.write_text(self.thread_id)

        # Retrieve API keys
        groq_api_key = BeagleEnv.get_env('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("Missing required GROQ_API_KEY in environment variables")
        
        # Initialize embeddings - use same model as process_repository.py
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},  # You can change to 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Database initialization with enhanced recovery
        self._initialize_vector_db("repository_content")
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=2000,  # Reduced to stay within rate limits
        )
        
        # Create the graph
        self.graph = self.create_graph()
        self.conversation_timeout = 3600  # 1 hour timeout for conversations
    
    def _initialize_vector_db(self, collection_name):
        """Initialize Milvus vector database with robust error handling"""
        try:
            self._connect_to_db(collection_name)
            return
        except Exception as e:
            logging.error(f"Milvus connection failed: {e}")
            raise ValueError(f"Milvus database initialization failed: {e}")
    
    def _connect_to_db(self, collection_name):
        """Establish connection to the Milvus vector database"""
        logging.info(f"Connecting to Milvus DB with collection: {collection_name}")
        
        # Connection parameters for Milvus
        connection_args = {
            "host": "localhost",  # Docker container host
            "port": "19530",      # Default Milvus port
        }
        
        self.vectordb = Milvus(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            consistency_level="Strong",
            vector_field="embedding",  # Match the field name from process_repository.py
            text_field="document",     # Match the document field name
        )
        
        logging.info("Successfully connected to Milvus vector database")
        
        # Create retriever
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={
                "k": 2,  # Retrieve only top 2 most relevant documents
            }
        )
    
    def retrieve_relevant_context(self, query: str) -> list[str]:
        """External method to retrieve context with enhanced error handling."""
        if not query or not query.strip():
            logging.warning("Empty query provided for context retrieval")
            return []
            
        try:
            logging.info(f"Retrieving context for query: {query[:50]}...")
            retrieved_docs = self.retriever.invoke(query)
            
            if not retrieved_docs:
                logging.info("No relevant documents found in vector database")
                return []
                
            logging.info(f"Retrieved {len(retrieved_docs)} relevant documents")
            return [doc.page_content for doc in retrieved_docs]
            
        except Exception as e:
            logging.error(f"Context retrieval error: {e}")
            if "connection" in str(e).lower() or "milvus" in str(e).lower():
                logging.warning("Milvus connection error detected. Attempting to reconnect...")
                try:
                    # Try to reconnect to Milvus using the stored collection name
                    self._connect_to_db(self.collection_name)
                    retrieved_docs = self.retriever.invoke(query)
                    return [doc.page_content for doc in retrieved_docs]
                except Exception as recovery_error:
                    logging.error(f"Failed to recover from Milvus connection error: {recovery_error}")
            return []

    def create_graph(self):
        """Create an improved LangGraph workflow for RAG."""
        def initialize_chat(state: ChatState):
            """Initialize the chat with system context while preserving history."""
            system_message = SystemMessage(content="""
                You are BeagleMind, the friendly and knowledgeable AI assistant for BeagleBoard. 
                Your personality traits:
                - Enthusiastic about embedded systems and single-board computers
                - Patient and encouraging with beginners
                - Technical but able to explain concepts clearly
                - Proactive in offering additional helpful information
                - Always positive and supportive
                - Maintain a conversational but professional tone
                                            
                Guidelines:
                1. Greet users warmly and introduce yourself at the start of conversations
                2. Answer questions accurately based on the BeagleBoard context
                3. If unsure, provide general guidance and suggest consulting documentation
                4. Never say "I don't know" - instead say "Let me check the resources..."
                5. Use emojis occasionally to make interactions friendlier 🐶, don't overdo this.
                6. Ask clarifying questions if requests are unclear
                7. Offer related information that might be helpful
                8. Maintain a conversational but professional tone
                9. If you get any irrelevant questions, answer as a normal LLM
                """)

            
            existing_messages = state.get("messages", [])
            existing_history = state.get("chat_history", [])
            user_messages = [msg for msg in existing_messages if isinstance(msg, HumanMessage)]
            
            initial_state = {
                "retrieved_context": state.get("retrieved_context", []),
                "chat_history": existing_history.copy()
            }
            
            if not existing_history:
                initial_state["messages"] = [system_message] + user_messages
            else:
                recent_history = existing_history[-2:]  # Only last 1 turn (reduced from 4)
                initial_state["messages"] = [system_message] + recent_history + user_messages
            
            return initial_state

        def retrieve_context(state: ChatState):
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                retrieved_context = self.retrieve_relevant_context(last_message.content)
                return {"retrieved_context": retrieved_context}
            return state
        
        def format_context_for_llm(state: ChatState):
            retrieved_context = state["retrieved_context"]
            last_message = state["messages"][-1]
            
            if retrieved_context:
                # Limit context size to prevent token overflow
                truncated_context = []
                for i, context in enumerate(retrieved_context):
                    # Limit each context chunk to 500 characters
                    truncated = context[:500] + "..." if len(context) > 500 else context
                    truncated_context.append(f"[Source {i+1}] {truncated}")
                
                context_text = "\n\n".join(truncated_context)
                augmented_message = HumanMessage(
                    content=f"Relevant Context:\n{context_text}\n\nQuery: {last_message.content}"
                )
                return {"messages": state["messages"][:-1] + [augmented_message]}
            return state
        
        def generate_response(state: ChatState):
            messages = state["messages"]
            response = self.llm.invoke(messages)
            return {
                "messages": state["messages"] + [response],
                "chat_history": state.get("chat_history", []) + [messages[-1], response],
                "retrieved_context": []
            }
        
        workflow = StateGraph(ChatState)
        workflow.add_node("initialize", initialize_chat)
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("format_context", format_context_for_llm)
        workflow.add_node("generate_response", generate_response)
        
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "retrieve_context")
        workflow.add_edge("retrieve_context", "format_context")
        workflow.add_edge("format_context", "generate_response")
        workflow.add_edge("generate_response", END)

        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def save_conversation_memory(self):
        try:
            serializable_memory = {
                thread_id: [serialize_message(msg) for msg in chat_history]
                for thread_id, chat_history in self.conversation_memory.items()
            }
            with open(self.memory_file, "w") as f:
                json.dump(serializable_memory, f)
        except Exception as e:
            logging.error(f"Error saving conversation memory: {e}")

    def invoke(self, message: str, thread_id: str = None):
        """Invoke with proper history maintenance and persistent memory saving."""
        if isinstance(message, str):
            message = HumanMessage(content=message)

        thread_id = thread_id or self.thread_id

        # Retrieve existing chat history for the thread, if any
        chat_history = self.conversation_memory.get(thread_id, [])
        chat_history.append(message)

        input_state = {
            "messages": chat_history,
            "retrieved_context": [],
            "chat_history": chat_history
        }

        try:
            response = self.graph.invoke(
                input_state,
                {"configurable": {"thread_id": thread_id}}
            )

            self.conversation_memory[thread_id] = response["chat_history"]
            self.save_conversation_memory()

            return {
                "response": response["messages"][-1].content,
                "thread_id": thread_id
            }

        except Exception as e:
            logging.error(f"Invocation error: {e}")
            raise

        
# CLI Implementation
@click.group()
def cli():
    """BeagleMind Chatbot CLI"""
    pass

_agent = None
_current_thread = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = BeagleMindAgent()
    return _agent

@cli.command()
@click.option('-p', '--prompt', required=True, help='User prompt for the chatbot')
@click.option('-l', '--log', help='Log file path')
@click.option('-t', '--thread', help='Thread ID for continuing conversation')
def chat(prompt, log, thread):
    """Interact with the chatbot"""
    try:
        global _current_thread
        
        chat_content = prompt
        if log:
            try:
                with open(log, 'r') as f:
                    chat_content += "\n" + f.read()
            except Exception as e:
                click.echo(f"Error reading log file: {e}")
        
        agent = get_agent()
        result = agent.invoke(chat_content, thread_id=thread or _current_thread)
        _current_thread = result["thread_id"]
        click.echo(f"BeagleMind: {result['response']}")
    
    except Exception as e:
        click.echo(f"Error during chat: {e}")

@cli.command()
@click.option('-g', '--generate', 'prompt', required=True, help='Generate a script (Python or Shell) based on user query')
@click.option('-o', '--output', help='Output file path for the generated script')
@click.option('-l', '--log', help='Log file path')
@click.option('-t', '--thread', help='Thread ID for continuing conversation')
@click.option('--type', 'script_type', type=click.Choice(['python', 'shell', 'auto'], case_sensitive=False), default='auto', help='Type of script to generate (default: auto-detect)')
def generate(prompt, output, log, thread, script_type):
    """Generate a script (Python or Shell) based on user query"""
    try:
        global _current_thread
        
        # Auto-detect script type if not specified
        detected_type = script_type
        if script_type == 'auto':
            # Check for shell-related keywords in the prompt
            shell_keywords = ['bash', 'shell', 'script', 'command', 'terminal', 'linux', 'grep', 'awk', 'sed', 'find', 'systemctl', 'service', 'cron', 'environment variable']
            python_keywords = ['python', 'import', 'function', 'class', 'pip', 'library', 'module']
            
            prompt_lower = prompt.lower()
            shell_score = sum(1 for keyword in shell_keywords if keyword in prompt_lower)
            python_score = sum(1 for keyword in python_keywords if keyword in prompt_lower)
            
            if shell_score > python_score:
                detected_type = 'shell'
            else:
                detected_type = 'python'
        
        # Construct the generation prompt based on script type
        if detected_type == 'shell':
            generation_prompt = f"""
You are a shell scripting expert. Generate a complete, working bash shell script that addresses this request:
{prompt}

Requirements:
1. Start with shebang: #!/bin/bash
2. Add error handling with set -e (exit on error)
3. Include helpful comments explaining each section
4. Use proper variable declarations and quoting
5. Include input validation where appropriate
6. Add usage information if the script takes arguments
7. Use appropriate exit codes
8. Make the script robust and production-ready

Generate ONLY the shell script code. Do not include any explanations, markdown formatting, or code blocks. Start directly with #!/bin/bash and provide only the script content.
"""
        else:
            generation_prompt = f"""
You are a Python programming expert. Generate a complete, working Python script that addresses this request:
{prompt}

Requirements:
1. Include proper shebang: #!/usr/bin/env python3
2. Add all necessary imports at the top
3. Include comprehensive error handling with try/except blocks
4. Add detailed comments explaining the code
5. Use proper function definitions and main() pattern
6. Include input validation where appropriate
7. Make the script executable and well-structured
8. Follow Python best practices (PEP 8)

Generate ONLY the Python script code. Do not include any explanations, markdown formatting, or code blocks. Start directly with #!/usr/bin/env python3 and provide only the script content.
"""
        
        # Add log content if provided
        if log:
            try:
                with open(log, 'r') as f:
                    generation_prompt += f"\n\nAdditional context from log:\n{f.read()}"
            except Exception as e:
                click.echo(f"Error reading log file: {e}")
        
        agent = get_agent()
        result = agent.invoke(generation_prompt, thread_id=thread or _current_thread)
        _current_thread = result["thread_id"]
        
        generated_code = result['response']
        
        # Debug: Show raw response for troubleshooting
        if not generated_code.strip():
            click.echo("Warning: Empty response from agent!")
            click.echo(f"Raw response: '{generated_code}'")
            return
        
        # Clean up any markdown formatting that might have slipped through
        original_code = generated_code
        
        # Remove markdown code blocks (more comprehensive)
        generated_code = re.sub(r'^```(?:python|bash|shell|sh)?\s*\n', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'^```(?:python|bash|shell|sh)?\s*', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'^```\s*\n', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'^```\s*', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'\n```\s*$', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'```\s*$', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'^```$', '', generated_code, flags=re.MULTILINE)
        
        # Remove any stray backticks at start/end of lines
        generated_code = re.sub(r'^`+', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'`+$', '', generated_code, flags=re.MULTILINE)
        
        # Remove filepath comments that might be added
        generated_code = re.sub(r'^# filepath:.*\n', '', generated_code, flags=re.MULTILINE)
        
        generated_code = generated_code.strip()
        
        # Debug: Show if cleanup changed anything
        if generated_code != original_code.strip():
            click.echo(f"Debug: Cleaned up markdown formatting (detected {detected_type} script)")
        
        # Final check for empty code
        if not generated_code:
            click.echo("Error: Generated code is empty after cleanup!")
            click.echo("Original response:")
            click.echo("=" * 30)
            click.echo(original_code)
            click.echo("=" * 30)
            return
        
        # Save to file if output path is specified
        if output:
            try:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Determine file extension based on script type
                if detected_type == 'shell':
                    if not output_path.suffix:
                        output_path = output_path.with_suffix('.sh')
                    elif output_path.suffix not in ['.sh', '.bash']:
                        output_path = output_path.with_suffix('.sh')
                else:  # python
                    if not output_path.suffix:
                        output_path = output_path.with_suffix('.py')
                    elif output_path.suffix != '.py':
                        output_path = output_path.with_suffix('.py')
                
                with open(output_path, 'w') as f:
                    f.write(generated_code)
                
                # Make the script executable
                output_path.chmod(0o755)
                
                script_type_name = "Shell" if detected_type == 'shell' else "Python"
                click.echo(f"{script_type_name} script generated and saved to: {output_path}")
                
                if detected_type == 'shell':
                    click.echo(f"You can run it with: ./{output_path} or bash {output_path}")
                else:
                    click.echo(f"You can run it with: python {output_path}")
                
            except Exception as e:
                click.echo(f"Error saving generated script: {e}")
                click.echo("Generated code:")
                click.echo(generated_code)
        else:
            # Display the generated code
            script_type_name = "Shell" if detected_type == 'shell' else "Python"
            click.echo(f"Generated {script_type_name} script:")
            click.echo("=" * 50)
            click.echo(generated_code)
            click.echo("=" * 50)
    
    except Exception as e:
        click.echo(f"Error during code generation: {e}")

@cli.command()
def init():
    """Initialize the chatbot"""
    try:
        load_dotenv()
        get_agent()
        click.echo("Chatbot initialized successfully!")
    except Exception as e:
        click.echo(f"Initialization error: {e}")

@cli.command()
def quit():
    """Quit the chatbot and reset global state"""
    global _agent, _current_thread
    _agent = None
    _current_thread = None
    click.echo("Chatbot memory reset.")

@cli.command()
def reset():
    """Reset the chatbot by quitting and reinitializing"""
    quit()
    init()

def main():
    cli()

if __name__ == '__main__':
    main()