# BeagleMind CLI 🐶💡

BeagleMind is an AI-powered Command Line Interface (CLI) Chatbot designed specifically to assist users with BeagleBoard documentation and embedded systems information.

## 🚀 Features

- Interactive AI chatbot for BeagleBoard-related queries
- Support for log file integration
- Simple command-line interface
- Contextual memory retention
- Easy initialization and reset functionality

## 📋 Prerequisites

- Python 3.7+
- pip
- Virtual environment (recommended)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/beaglemind-cli.git
cd beaglemind-cli
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```
GROQ_API_KEY=your_groq_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_KEY=your_azure_openai_key
```

## 💻 Usage

### Chat with BeagleMind

Send a prompt to the chatbot:
```bash
beaglemind chat -p "What is a BeagleBone?"
```

Chat with a log file:
```bash
beaglemind chat -p "Explain embedded systems" -l conversation.log
```

### Other Commands

Initialize the chatbot:
```bash
beaglemind init
```

Quit and erase chatbot memory:
```bash
beaglemind quit
```

Reset the chatbot:
```bash
beaglemind reset
```

## 🛠 Configuration

- The chatbot uses Groq's Gemma2 9B model
- Vector database is loaded from `src/data/vectordb`
- Embeddings are generated using Azure OpenAI

## 📄 Logging

- Log files must have a `.log` extension
- Logs can be integrated with chat prompts for context

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📬 Support

For issues, questions, or suggestions, please [open an issue](https://github.com/yourusername/beaglemind-cli/issues) on GitHub.

## 📃 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙌 Acknowledgments

- [BeagleBoard](https://beagleboard.org/)
- [Langchain](https://www.langchain.com/)
- [Groq](https://groq.com/)