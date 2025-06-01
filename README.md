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
- Generate Python and shell scripts based on user requirements
- Provide technical guidance and troubleshooting help
- Maintain conversation context through persistent memory
- Leverage vector databases for retrieving relevant documentation

## Installation

### Prerequisites

Before installation, ensure you have:
- Python 3.9 or higher
- pip (Python package manager)
- Groq API key (for LLM access)
- Docker with Milvus vector database (for RAG capabilities)

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
```

> **Important:** A valid `.env` file with a Groq API key must be present in the directory where you run the BeagleMind CLI.

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

### Generate Scripts

BeagleMind can intelligently generate Python and shell scripts based on your requirements. The CLI automatically detects the appropriate script type based on keywords in your prompt, or you can specify it manually.

##### Features:
- **Automatic script type detection** based on prompt content
- **Production-ready code** with error handling and best practices
- **Executable scripts** with proper shebangs and permissions
- **Comprehensive documentation** with inline comments
- **File output** with automatic extension handling

##### Auto-detect Script Type

BeagleMind analyzes your prompt and automatically determines whether to generate a Python or shell script:

```bash
# This will generate a Python script (detects 'python', 'import', etc.)
beaglemind generate -g "Create a Python script to read temperature from sensors"

# This will generate a shell script (detects 'bash', 'command', 'terminal', etc.)
beaglemind generate -g "Create a bash script to install dependencies"
```

##### Specify Script Type

You can explicitly specify the script type:

```bash
# Force Python script generation
beaglemind generate -g "Create a monitoring tool" --type python

# Force shell script generation  
beaglemind generate -g "Create a backup utility" --type shell
```

##### Save to File

Generate and save scripts directly to files with automatic extension handling:

```bash
# Save Python script (automatically adds .py extension)
beaglemind generate -g "Create a GPIO control script" -o gpio_control

# Save shell script (automatically adds .sh extension)
beaglemind generate -g "Create system setup script" -o setup_system

# Specify full filename
beaglemind generate -g "Create monitoring script" -o monitor.py
```

##### Include Log Context

Include content from log files to help with debugging and script generation:

```bash
# Generate a script to fix errors found in logs
beaglemind generate -g "Create a script to fix this error" -l error.log -o fix_error.sh

# Generate diagnostic script based on system logs
beaglemind generate -g "Create diagnostic script for these issues" -l system.log
```

##### Continue in Conversation Thread

Use thread IDs to continue script development in context:

```bash
# Initial script generation
beaglemind generate -g "Create a backup script" -t my_session

# Modify the script in the same context
beaglemind generate -g "Add email notifications to the previous script" -t my_session

# Further improvements
beaglemind generate -g "Add compression and encryption" -t my_session
```

##### Example Outputs

**Python Script Example:**
```bash
beaglemind generate -g "Create a script to monitor BeagleBoard GPIO pins" -o gpio_monitor.py
```

Generated script features:
- Proper shebang (`#!/usr/bin/env python3`)
- Comprehensive imports
- Error handling with try/except blocks
- Function definitions and main() pattern
- Input validation
- Detailed comments
- PEP 8 compliance

**Shell Script Example:**
```bash
beaglemind generate -g "Create a setup script for BeagleBoard development environment" -o setup_dev.sh
```

Generated script features:
- Proper shebang (`#!/bin/bash`)
- Error handling with `set -e`
- Variable declarations and quoting
- Input validation
- Usage information
- Exit codes
- Detailed comments
