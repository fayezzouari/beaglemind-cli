# BeagleMind CLI

<div align="center">
  <img src="beaglemind/assets/logo.png" width="100">
</div>

BeagleMind is an intelligent AI assistant specifically tailored for BeagleBoard embedded systems and single-board computing enthusiasts. It combines powerful language models with retrieval-augmented generation (RAG) capabilities to provide knowledgeable, context-aware responses.

> **Note:** This CLI tool is a proof of concept for a BeagleBoard Google Summer of Code (GSoC) project. It aims to deliver and concretize the concept of an AI assistant specifically trained for BeagleBoard systems.

## Project Vision

While the current implementation uses third-party LLM providers (Groq), the final vision is to host a fine-tuned language model directly on BeagleBoard hardware, specifically trained on BeagleBoard documentation and community data. This will create a truly specialized AI assistant for the BeagleBoard ecosystem.

> **Note:** This CLI is currently under active development. Features and APIs are subject to change.

## Overview

BeagleMind offers a command-line interface for interacting with an AI assistant that can:
- Answer questions about BeagleBoard products and embedded systems
- Provide technical guidance and troubleshooting help
- Maintain conversation context through persistent memory
- Leverage vector databases for retrieving relevant documentation

## Installation

### Prerequisites

Before installation, ensure you have:
- Python 3.9 or higher
- pip (Python package manager)
- Groq API key (for LLM access)
- Jina AI API key (for embeddings)

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/fayezzouari/beaglemind-cli.git
cd beaglemind-cli
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Create a `.env` file in your project directory with the following content:

```
GROQ_API_KEY=your_groq_api_key_here
JINA_API_KEY=your_jina_api_key_here
```

> **Important:** A valid `.env` file with API keys must be present in the directory where you run the BeagleMind CLI.

### Installation Methods

#### Method 1: Development Install (Recommended during development)

For development purposes, install the package in editable mode:

```bash
pip install -e .
```

#### Method 2: Direct Installation

Install from the repository:

```bash
pip install .
```

## Usage

BeagleMind CLI provides several commands for interaction:

### Initialize the Agent

Before first use, initialize the agent:

```bash
beaglemind init
```

<div align="center">
  <img src="beaglemind/assets/init.gif">
</div>

### Chat with BeagleMind

Send a prompt to BeagleMind:

```bash
beaglemind chat -p "Hi, What is BeagleBoard?"
```

<div align="center">
  <img src="beaglemind/assets/chat.gif">
</div>
### Continue a Conversation

Use the thread ID to continue a previous conversation:

```bash
beaglemind chat -p "How do I connect it to my computer?" -t your_thread_id
```

### Include Content from a File

Include content from a log file in your prompt:

```bash
beaglemind chat -p "Debug this error" -l error_log.txt
```

> **Note:** The logs feature is still under development and maintenance. It may not function optimally in all scenarios.

### Reset or Quit

Reset the agent's memory:

```bash
beaglemind reset
```

Quit and clear the agent's state:

```bash
beaglemind quit
```

<div align="center">
  <img src="beaglemind/assets/quit.gif">
</div>

## Project Structure

```
beaglemind-cli/
├── beaglemind/
│   ├── __init__.py
│   ├── main.py              # Main CLI implementation
│   ├── beagleenv.py         # Environment variable management
│   ├── db_repair.py         # Vector database repair utilities
│   ├── embedding.py         # Embedding model implementations
│   ├── .env                    # API keys and configuration (create this)
│   └── data/
│       └── vectordb/        # Vector database storage 
├── requirements.txt        # Package dependencies
├── setup.py                # Package setup file
└── README.md               # This documentation
```

### API Key Errors

If you see errors related to missing API keys:

1. Verify your `.env` file exists in the current directory
2. Check that both `GROQ_API_KEY` and `JINA_API_KEY` are set correctly
3. Restart the CLI application after updating the `.env` file


## GSoC Project Status

This project is part of a Google Summer of Code initiative for BeagleBoard. Current development focuses on:

1. Establishing the RAG architecture with vector databases
2. Creating an intuitive CLI interface
3. Laying groundwork for future fine-tuning on BeagleBoard-specific data
4. Developing the concept for eventual fine-tuned LLM hosting

## Contact

For questions or issues, please contact Fayez Zouari at fayez.zouari@insat.ucar.tn.