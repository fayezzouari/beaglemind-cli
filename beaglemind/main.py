import warnings
warnings.simplefilter("ignore")  # Suppress ALL warnings (not recommended long-term)

import os
import logging
import click
from pathlib import Path
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

# Other necessary imports
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from beaglemind.embedding import AzureOpenAIEmbeddings
from beaglemind.beagleenv import BeagleEnv
class ChatState(TypedDict):
    """Enhanced state for the chatbot with more precise tracking."""
    messages: list[AnyMessage]
    vectordb: Chroma
    retriever: object
    retrieved_context: list[Document]
    chat_history: list[AnyMessage]


import warnings
warnings.simplefilter("ignore")  # Suppress ALL warnings (not recommended long-term)

import os
import logging
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
from beaglemind.embedding import AzureOpenAIEmbeddings
from beaglemind.beagleenv import BeagleEnv

# Disable PostHog tracking
import os
os.environ['POSTHOG_DISABLE_SEND'] = 'true'

class ChatState(TypedDict):
    """Enhanced state for the chatbot with more precise tracking."""
    messages: list[AnyMessage]
    retrieved_context: list[str]
    chat_history: list[AnyMessage]

class BeagleMindAgent:
    def __init__(self, vectordb_path="data/vectordb"):
        """Initialize the BeagleMind agent with improved LangGraph RAG."""
        load_dotenv()
        
        # API Key validation
        BeagleEnv.load_env_file()
        
        # Retrieve API keys
        groq_api_key = BeagleEnv.get_env('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("Missing required API keys in environment variables")
        
        # Initialize embeddings and vector database
        self.embeddings = AzureOpenAIEmbeddings()
        self.vectordb = Chroma(
            persist_directory=vectordb_path,
            embedding_function=self.embeddings
        )
        
        # Create retriever with specific configuration
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={
                "k": 10,  # Retrieve top 5 most relevant documents
                "score_threshold": 0.5  # Only retrieve documents above relevance threshold
            }
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it",
            temperature=0.3,
        )
        
        # Create the graph
        self.graph = self.create_graph()
    
    def retrieve_relevant_context(self, query: str) -> list[str]:
        """External method to retrieve context, separated from graph state."""
        try:
            retrieved_docs = self.retriever.invoke(query)
            return [doc.page_content for doc in retrieved_docs]
        except Exception as e:
            logging.error(f"Context retrieval error: {e}")
            return []
    
    def create_graph(self):
        """Create an improved LangGraph workflow for RAG."""
        def initialize_chat(state: ChatState):
            """Initialize the chat with system context."""
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
            
            return {
                "messages": [system_message],
                "retrieved_context": [],
                "chat_history": []
            }
        
        def retrieve_context(state: ChatState):
            """Advanced context retrieval with relevance filtering."""
            last_message = state["messages"][-1]
            
            if isinstance(last_message, HumanMessage):
                # Use external method to retrieve context
                retrieved_context = self.retrieve_relevant_context(last_message.content)
                
                return {
                    "retrieved_context": retrieved_context,
                }
            return state
        
        def format_context_for_llm(state: ChatState):
            """Prepare retrieved context for LLM input."""
            retrieved_context = state["retrieved_context"]
            last_message = state["messages"][-1]
            
            if retrieved_context:
                # Create a context string with source references
                context_text = "\n\n".join([
                    f"[Source {i+1}] {context}"
                    for i, context in enumerate(retrieved_context)
                ])
                
                # Augment the original message with retrieved context
                augmented_message = HumanMessage(
                    content=f"Relevant Context:\n{context_text}\n\nQuery: {last_message.content}"
                )
                
                return {
                    "messages": state["messages"][:-1] + [augmented_message],
                }
            return state
        
        def generate_response(state: ChatState):
            """Generate a response using the LLM with context."""
            messages = state["messages"]
            
            # Invoke LLM with enhanced system and context-aware prompting
            response = self.llm.invoke(messages)
            
            return {
                "messages": state["messages"] + [response],
                "chat_history": state.get("chat_history", []) + [messages[-1], response],
                # Reset retrieved context to prepare for next interaction
                "retrieved_context": []
            }
        
        # Define the graph structure
        workflow = StateGraph(ChatState)
        
        # Add nodes with improved workflow
        workflow.add_node("initialize", initialize_chat)
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("format_context", format_context_for_llm)
        workflow.add_node("generate_response", generate_response)
        
        # Define more precise workflow edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "retrieve_context")
        workflow.add_edge("retrieve_context", "format_context")
        workflow.add_edge("format_context", "generate_response")
        
        # Use memory checkpointing
        checkpointer = MemorySaver()
        
        # Compile the graph
        return workflow.compile(checkpointer=checkpointer)
    
    def invoke(self, message: str):
        """Invoke the chatbot with a message."""
        # If message is a string, convert to HumanMessage
        if isinstance(message, str):
            message = HumanMessage(content=message)
        
        # Stream the response
        response = self.graph.invoke(
            {"messages": [message]},
            {"configurable": {"thread_id": "unique_conversation"}}
        )
        
        # Return the last AI message
        return response["messages"][-1].content




# CLI Implementation
@click.group()
def cli():
    """BeagleMind Chatbot CLI"""
    pass

# Global agent instance
_agent = None

def get_agent():
    """Singleton pattern for agent initialization"""
    global _agent
    if _agent is None:
        _agent = BeagleMindAgent()
    return _agent

@cli.command()
@click.option('-p', '--prompt', required=True, help='User prompt for the chatbot')
@click.option('-l', '--log', help='Log file path')
def chat(prompt, log):
    """Interact with the chatbot"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Combine prompt with log content if log file is provided
        chat_content = prompt
        if log:
            try:
                with open(log, 'r') as f:
                    chat_content += "\n" + f.read()
            except Exception as e:
                click.echo(f"Error reading log file: {e}")
        
        # Get the agent and invoke chat
        agent = get_agent()
        response = agent.invoke(chat_content)
        click.echo(f"BeagleMind: {response}")
    
    except Exception as e:
        click.echo(f"Error during chat: {e}")

@cli.command()
def init():
    """Initialize the chatbot"""
    try:
        load_dotenv()
        get_agent()  # This will initialize the agent
        click.echo("Chatbot initialized successfully!")
    except Exception as e:
        click.echo(f"Initialization error: {e}")

@cli.command()
def quit():
    """Quit the chatbot and reset global state"""
    global _agent
    _agent = None
    click.echo("Chatbot memory reset.")

@cli.command()
def reset():
    """Reset the chatbot by quitting and reinitializing"""
    quit()
    init()

def main():
    """Entry point for the CLI application"""
    cli()

if __name__ == '__main__':
    main()
