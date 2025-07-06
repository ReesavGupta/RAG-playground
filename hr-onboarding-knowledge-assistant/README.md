# 🧑‍💼 HR Onboarding Knowledge Assistant

A comprehensive RAG (Retrieval-Augmented Generation) system designed to provide intelligent HR support and onboarding assistance. This system analyzes company documentation, HR policies, and onboarding materials to deliver contextually relevant answers to employee questions.

## ✨ Features

- **📚 Knowledge Base Integration**: Processes PDF and Word documents containing HR policies and onboarding materials
- **🤖 AI-Powered Q&A**: Intelligent response generation using Ollama LLM (Llama 3:8B)
- **🔍 Semantic Search**: Advanced vector search for finding relevant policy information
- **📊 Source Attribution**: Provides detailed source references for all answers
- **🌐 Web Interface**: User-friendly Streamlit application for easy interaction
- **📝 Document Processing**: Automatic ingestion and chunking of HR documentation

## 🏗️ Architecture

```
HR Documents → Document Loading → Text Chunking → Vector Storage → Semantic Search → AI Response Generation
```

### Core Components

- **`ingest.py`**: Handles document loading, processing, and vector database creation
- **`qa_chain.py`**: Builds the RAG chain with Ollama LLM and custom prompts
- **`config.py`**: Centralized configuration for models and paths
- **`app.py`**: Streamlit web interface for user interaction
- **`main.py`**: Simple entry point for the application

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Ollama with Llama 3:8B model installed
- HR documentation files (PDF/DOCX) in the `data/` directory

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hr-onboarding-knowledge-assistant
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```

3. **Install Ollama and Llama 3:8B**
   ```bash
   # Install Ollama (follow instructions at https://ollama.ai)
   # Then pull the Llama 3:8B model
   ollama pull llama3:8b
   ```

4. **Prepare your HR documentation**
   ```bash
   # Place your HR documents in the data/ directory
   # Supported formats: PDF, DOCX
   ```

### Usage

#### 1. Ingest HR Documents

```bash
python ingest.py
```

This will:
- Load all PDF and DOCX files from the `data/` directory
- Split documents into manageable chunks (800 chars with 100 char overlap)
- Create embeddings using HuggingFace sentence transformers
- Store in Chroma vector database

#### 2. Run the Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and start asking HR questions!

## 📁 Project Structure

```
hr-onboarding-knowledge-assistant/
├── app.py                 # Streamlit web interface
├── ingest.py              # Document ingestion and processing
├── qa_chain.py            # RAG chain implementation
├── config.py              # Configuration settings
├── main.py                # Simple entry point
├── data/                  # HR documentation files
│   ├── Major Project Final Report.pdf
│   └── Untitled document.pdf
├── vectorstore/           # Vector database storage
│   ├── chroma.sqlite3
│   └── a6cb483b-76a5-4ca6-9d69-ac0ee0cce1ef/
├── pyproject.toml         # Project dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Dependencies

The project uses the following key dependencies:

- **`langchain`**: RAG framework
- **`langchain-chroma`**: Vector database integration
- **`langchain-huggingface`**: Embeddings using sentence-transformers
- **`langchain-ollama`**: Local LLM integration
- **`langchain-community`**: Document loaders for PDF/DOCX
- **`pypdf`**: PDF processing
- **`streamlit`**: Web interface
- **`sentence-transformers`**: Text embeddings

### Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace)
- **LLM**: `llama3:8b` (via Ollama)
- **Vector Database**: Chroma with similarity search
- **Chunk Size**: 800 characters with 100 character overlap

## 💡 How It Works

### 1. Document Processing
- Load PDF and DOCX files from the `data/` directory
- Extract text content using appropriate loaders
- Split documents into manageable chunks for processing

### 2. Vector Storage
- Generate embeddings using HuggingFace sentence transformers
- Store chunks in Chroma vector database with metadata
- Enable semantic search capabilities

### 3. Query Processing
- Receive user questions through Streamlit interface
- Perform semantic search to find relevant document chunks
- Retrieve top 4 most similar chunks

### 4. Response Generation
- Use Ollama LLM with custom prompt template
- Generate contextually relevant answers
- Provide source attribution for transparency

## 🎯 Example Usage

### Web Interface
1. Start the Streamlit app: `streamlit run app.py`
2. Type your HR question in the text input
3. Click "Get Answer" to receive AI-generated response
4. Review source documents for verification

### Programmatic Usage
```python
from qa_chain import get_qa_chain

# Initialize the QA chain
qa_chain = get_qa_chain()

# Ask a question
query = "What is the company's vacation policy?"
result = qa_chain.invoke({"query": query})

print("Answer:", result["result"])
print("Sources:", [doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]])
```

## 🌐 Web Interface Features

The Streamlit app provides:

- **Simple Query Interface**: Type HR questions in natural language
- **Real-time Answers**: Get instant responses with AI
- **Source Attribution**: See which documents were referenced
- **Clean UI**: Professional interface for HR interactions

## 🔍 Advanced Features

### Document Processing
- **Multi-format Support**: PDF and DOCX files
- **Intelligent Chunking**: Preserves document structure
- **Metadata Preservation**: Maintains source information

### Semantic Search
- **Similarity-based Retrieval**: Finds most relevant content
- **Configurable Results**: Adjustable number of retrieved chunks
- **Context Preservation**: Maintains document context

### Response Generation
- **Custom Prompts**: Tailored for HR-specific responses
- **Source Attribution**: Transparent answer sources
- **Confidence Scoring**: Built into the LLM response

## 🛠️ Customization

### Changing Models
```python
# In config.py
EMBEDDING_MODEL = "your-embedding-model"
OLLAMA_MODEL = "your-llm-model"
```

### Adjusting Chunk Size
```python
# In ingest.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase chunk size
    chunk_overlap=200,  # Adjust overlap
    # ... other parameters
)
```

### Modifying Search Parameters
```python
# In qa_chain.py
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # More results
```

### Custom Prompts
```python
# In qa_chain.py
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an HR assistant. Use the following HR policy context to answer employee questions accurately and professionally.

    Context:
    {context}

    Question: {question}
    Answer:
    """
)
```

## 📊 Sample Use Cases

### HR Policy Questions
- "What is the company's remote work policy?"
- "How many vacation days do I get?"
- "What is the process for requesting time off?"
- "What are the benefits for new employees?"

### Onboarding Queries
- "What documents do I need for onboarding?"
- "How do I set up my benefits?"
- "What is the dress code policy?"
- "How do I access company resources?"

### Employee Support
- "What is the grievance procedure?"
- "How do I report harassment?"
- "What are the working hours?"
- "How do I request accommodations?"

## 🔧 Technical Implementation

### RAG Pipeline
1. **Document Ingestion**: Load and process HR documents
2. **Text Chunking**: Split into searchable segments
3. **Vector Embedding**: Create semantic representations
4. **Storage**: Store in Chroma vector database
5. **Retrieval**: Semantic search for relevant chunks
6. **Generation**: LLM-based answer synthesis

### Similarity Matching
- Uses cosine similarity for document retrieval
- Configurable number of top results (default: 4)
- Maintains document metadata for source attribution

### Response Generation
- Custom prompt template for HR context
- Temperature setting for response creativity
- Source document integration for accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ollama for local LLM capabilities
- Chroma for vector database
- Streamlit for the web interface
- LangChain for the RAG framework
- HuggingFace for embedding models

---

**Empowering HR with AI! 🧑‍💼✨**
