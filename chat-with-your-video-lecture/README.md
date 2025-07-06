# 🎓 Chat with Your Video Lecture

A powerful RAG (Retrieval-Augmented Generation) application that allows you to interact with video lectures through natural language queries. This project transcribes video lectures, chunks them with precise timestamps, stores them in a vector database, and provides an intelligent Q&A interface.

## ✨ Features

- **🎬 Video Transcription**: Extract audio from video files and transcribe using OpenAI Whisper
- **⏱️ Timestamp Preservation**: Maintain precise timestamp information for each text chunk
- **🔍 Vector Search**: Store transcriptions in Chroma vector database with semantic search capabilities
- **🤖 AI-Powered Q&A**: Query your lecture content using Ollama LLM (Llama 3:8B)
- **🌐 Web Interface**: Interactive Streamlit web application for easy interaction
- **📊 Source Attribution**: Get detailed source segments with timestamps for all answers

## 🏗️ Architecture

```
Video Lecture → Audio Extraction → Transcription → Chunking → Vector Storage → Q&A Interface
```

### Core Components

- **`transcriber.py`**: Handles video-to-audio conversion and transcription
- **`chunker.py`**: Splits transcript into chunks while preserving timestamp metadata
- **`vector_stoer.py`**: Manages Chroma vector database operations
- **`qa_chain.py`**: Builds the RAG chain with Ollama LLM
- **`utils.py`**: Utility functions for timestamp formatting and source summarization
- **`streamlit_app.py`**: Web interface for user interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg installed on your system
- Ollama with Llama 3:8B model installed

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chat-with-your-video-lecture
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

### Usage

#### 1. Process a Video Lecture

```python
from app import main

# Place your video file in sample-data/ directory
# Update video_path in app.py if needed
main()
```

#### 2. Run the Web Interface

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501` and start asking questions about your lecture!

## 📁 Project Structure

```
chat-with-your-video-lecture/
├── app.py                 # Main application entry point
├── streamlit_app.py       # Web interface
├── transcriber.py         # Video transcription utilities
├── chunker.py            # Text chunking with timestamps
├── vector_stoer.py       # Vector database operations
├── qa_chain.py           # RAG chain implementation
├── utils.py              # Utility functions
├── main.py               # Simple entry point
├── sample-data/          # Sample video files
│   └── videoplayback.mp4
├── index/                # Vector database storage
├── pyproject.toml        # Project dependencies
└── README.md            # This file
```

## 🔧 Configuration

### Dependencies

The project uses the following key dependencies:

- **`openai-whisper`**: For video transcription
- **`ffmpeg-python`**: For audio extraction from video
- **`langchain`**: RAG framework
- **`langchain-chroma`**: Vector database integration
- **`langchain-huggingface`**: Embeddings using sentence-transformers
- **`langchain-ollama`**: Local LLM integration
- **`streamlit`**: Web interface
- **`sentence-transformers`**: Text embeddings

### Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace)
- **LLM**: `llama3:8b` (via Ollama)
- **Vector Database**: Chroma with similarity search

## 💡 How It Works

### 1. Video Processing
- Extract audio from video files using FFmpeg
- Transcribe audio using OpenAI Whisper
- Clean up temporary audio files

### 2. Text Chunking
- Split transcript into manageable chunks (default: 1000 chars)
- Preserve precise timestamp information for each chunk
- Create character-to-timestamp mapping for accurate time tracking

### 3. Vector Storage
- Generate embeddings using HuggingFace sentence transformers
- Store chunks in Chroma vector database
- Enable semantic search capabilities

### 4. Q&A Interface
- Build RAG chain with Ollama LLM
- Retrieve relevant chunks based on query similarity
- Generate answers with source attribution
- Display timestamps for source segments

## 🎯 Example Usage

```python
# Process a video lecture
video_path = "sample-data/lecture.mp4"
audio_path = extract_audio(video_path)
segments = transcribe(audio_path)
chunks = chunk_transcript_with_timestamps(segments)
db = store_embedding(chunks)

# Ask questions
qa_chain = build_qa_chain(db)
query = "What are the main topics discussed in the first 30 minutes?"
response = qa_chain({"query": query})

print("Answer:", response["result"])
print("Sources:", summarize_sources(response["source_documents"]))
```

## 🌐 Web Interface Features

The Streamlit app provides:

- **Simple Query Interface**: Type questions about your lecture
- **Real-time Answers**: Get instant responses with AI
- **Source Attribution**: See which parts of the lecture were referenced
- **Timestamp Display**: Navigate directly to relevant video segments

## 🔍 Advanced Features

### Timestamp Precision
The chunking system maintains precise timestamp information:
- Character-level timestamp mapping
- Linear interpolation within segments
- Accurate start/end times for each chunk

### Source Attribution
Every answer includes:
- Source document chunks
- Start and end timestamps
- Duration information
- Character position mapping

### Vector Search
- Similarity-based retrieval
- Configurable number of results (default: 4)
- Semantic understanding of queries

## 🛠️ Customization

### Changing Models
```python
# In qa_chain.py
llm=OllamaLLM(model="your-model", temperature=0)

# In vector_stoer.py
EMBEDDING_MODEL = "your-embedding-model"
```

### Adjusting Chunk Size
```python
# In chunker.py
chunk_transcript_with_timestamps(segments, chunk_size=1500, chunk_overlap=300)
```

### Modifying Search Parameters
```python
# In qa_chain.py
retriever = vectorstore.as_retriever(search_type="similarity", k=6)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for Whisper transcription model
- Ollama for local LLM capabilities
- Chroma for vector database
- Streamlit for the web interface
- LangChain for the RAG framework

---

**Happy Learning! 🎓✨**
