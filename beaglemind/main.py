import datetime
import json
import os
import logging
import uuid
import click
from pathlib import Path
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# Other necessary imports
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
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
    def __init__(self, vectordb_path="data/vectordb", memory_file="conversation_memory.json"):
        """Initialize the BeagleMind agent with improved database handling."""
        load_dotenv()
        
        # API Key validation
        BeagleEnv.load_env_file()
        self.memory_file = Path(memory_file)
        if self.memory_file.exists():
            try:
                # Read the file content and check if it's empty
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

        # Optionally, load or set a persistent thread ID.
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
        
        # Initialize embeddings
        jina_api_key = BeagleEnv.get_env('JINA_API_KEY')
        if not jina_api_key:
            raise ValueError("Missing required JINA_API_KEY in environment variables")
            
        self.embeddings = AzureOpenAIEmbeddings()
        
        # Database initialization with enhanced recovery
        db_path = Path(vectordb_path)
        self._initialize_vector_db(db_path)
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it",
            temperature=0.3,
        )
        
        # Create the graph
        self.graph = self.create_graph()
        self.conversation_timeout = 3600  # 1 hour timeout for conversations
    
    def _initialize_vector_db(self, db_path):
        """Initialize vector database with robust error handling and recovery"""
        sqlite_file = db_path / "chroma.sqlite3"
        
        # First, attempt to validate the existing database
        if VectorDBRepair.validate_database(db_path):
            try:
                self._connect_to_db(db_path)
                return
            except Exception as e:
                logging.error(f"Database validation passed but connection failed: {e}")
                # Continue to repair
        
        # If we reached here, the database needs repair
        logging.info("Attempting database repair...")
        repair_success = VectorDBRepair.repair_database(db_path)
        
        if repair_success:
            try:
                self._connect_to_db(db_path)
                return
            except Exception as e:
                logging.error(f"Database repair succeeded but connection still failed: {e}")
                # Last resort: clean install
        
        # If repair didn't work, try clean install
        logging.warning("Attempting clean database installation...")
        clean_success = VectorDBRepair.clean_install(db_path)
        
        if clean_success:
            try:
                self._connect_to_db(db_path)
                return
            except Exception as e:
                logging.error(f"Clean installation failed: {e}")
                raise ValueError(f"Vector database initialization failed after all recovery attempts: {e}")
        else:
            raise ValueError("Failed to initialize vector database. All recovery attempts failed.")
    
    def _connect_to_db(self, db_path):
        """Establish connection to the vector database"""
        logging.info(f"Connecting to Chroma DB at {db_path}")
        self.vectordb = Chroma(
            persist_directory=str(db_path),
            embedding_function=self.embeddings,
        )
        
        # Verify connection with a test query
        self.vectordb.get(include=[], limit=1)
        logging.info("Successfully connected to vector database")
        
        # Create retriever
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={
                "k": 10,  # Retrieve top 10 most relevant documents
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
            if "file is not a database" in str(e):
                logging.warning("Database error detected. Attempting to reinitialize...")
                try:
                    db_path = Path(self.vectordb._persist_directory)
                    sqlite_file = db_path / "chroma.sqlite3"
                    if sqlite_file.exists():
                        sqlite_file.unlink()
                        
                    self.vectordb = Chroma(
                        persist_directory=str(db_path),
                        embedding_function=self.embeddings,
                    )
                    self.retriever = self.vectordb.as_retriever(
                        search_kwargs={"k": 10}
                    )
                    retrieved_docs = self.retriever.invoke(query)
                    return [doc.page_content for doc in retrieved_docs]
                except Exception as recovery_error:
                    logging.error(f"Failed to recover from database error: {recovery_error}")
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
                recent_history = existing_history[-4:]  # Last 2 turns
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
                context_text = "\n\n".join([
                    f"[Source {i+1}] {context}"
                    for i, context in enumerate(retrieved_context)
                ])
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