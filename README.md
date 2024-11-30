# CodebaseRag - Multi-Repository Code Search Assistant

A Streamlit-powered RAG (Retrieval-Augmented Generation) application that enables users to search and query across multiple codebases. The app features code screenshot analysis capabilities and provides context-aware responses using Pinecone for vector search.

## Features

- Search across multiple GitHub repositories simultaneously
- Upload and analyze code screenshots using OCR
- RAG-powered responses using Pinecone vector database
- Interactive chat interface
- Support for switching between different codebases
- OCR capability for code screenshots

## Prerequisites

- Python 3.8+
- Tesseract OCR
- API keys for:
  - Pinecone
  - Groq
- GitHub repositories indexed in Pinecone

## Installation

1. Install Tesseract OCR:
```bash
# Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your-key
GROQ_API_KEY=your-key
PINECONE_API_KEY=your-key
PINECONE_ENV=us-east-1
PINECONE_INDEX_SECURE=codebase1-rag
PINECONE_INDEX_CHATBOT=codebase2-rag
```

2. Create a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "your-key"
GROQ_API_KEY = "your-key"
PINECONE_API_KEY = "your-key"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_SECURE = "codebase1-rag"
PINECONE_INDEX_CHATBOT = "codebase2-rag"
```

## Running Locally

1. Activate your virtual environment (if using one)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Run the Streamlit app:
```bash
streamlit run chat.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Select one or both codebases to search from the available options
2. (Optional) Upload a code screenshot for specific code-related questions
3. Enter your query in the chat input
4. View the AI-powered response based on the selected codebases and uploaded screenshot

## Deploying to Streamlit Cloud

1. Fork this repository
2. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add your secrets in the Streamlit Cloud dashboard:
   - Go to App Settings > Secrets
   - Add all the required API keys and configuration values
5. Deploy the app

## Project Structure

```
.
├── chat.py              # Main Streamlit application
├── embeddings.py        # Embedding generation utilities
├── rag_utils.py         # RAG search and processing functions
├── requirements.txt     # Python dependencies
└── packages.txt         # System dependencies (Tesseract)
------.env# not included, add this yourself.
------.streamlit/secrets.toml # not inlcuded, add this yourself.
```

## Limitations

- Requires pre-indexed repositories in Pinecone
- OCR quality depends on screenshot clarity
- API keys required for full functionality



