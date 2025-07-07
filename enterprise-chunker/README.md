# Enterprise Chunker: Intelligent Document Chunking for Knowledge Management

## Problem Statement

Enterprise software companies manage vast and diverse knowledge bases containing technical documentation, support tickets, API references, policies, and tutorials. Traditional uniform chunking methods often:
- Break code snippets
- Separate troubleshooting steps
- Disconnect policy requirements

This leads to poor retrieval accuracy and suboptimal knowledge management for internal teams and support automation.

## Challenge

**Build an adaptive chunking system that automatically detects document types and applies appropriate chunking strategies (semantic, code-aware, hierarchical) to improve knowledge retrieval for internal teams and support automation.**

## Solution Approach

1. **Document Classification**: Automatically detect content types and structure patterns in documents.
2. **Adaptive Chunking**: Apply document-specific chunking strategies for optimal context preservation (e.g., semantic, code-aware, hierarchical chunking).
3. **LangChain Integration**: Use LangChain to orchestrate the processing pipeline and manage the vector store (ChromaDB).
4. **Performance Monitoring**: Track retrieval accuracy and refine chunking strategies based on evaluation metrics.

## Key Inputs

- Mixed enterprise documents (e.g., Confluence, Jira, GitHub wikis, PDFs)
- Document metadata and usage patterns
- User query success metrics

## Expected Outputs

- Optimally chunked document collections
- Improved retrieval accuracy metrics (precision, recall, F1)
- Automated processing pipeline for new and updated content

## Features

- **Automatic Document Type Detection**: Classifies documents (API reference, policy, troubleshooting guide, support ticket, etc.)
- **Adaptive Chunking**: Applies the best chunking strategy for each document type
- **Vector Store Integration**: Stores chunks in ChromaDB for efficient retrieval
- **Retrieval Evaluation**: Measures retrieval performance with precision, recall, and F1 metrics
- **Interactive Demo**: Run end-to-end processing, evaluation, and search from a single script

## Project Structure

```
enterprise-chunker/
  ├── main.py                # Core logic and EnterpriseKnowledgeManager
  ├── main_demo.py           # End-to-end demo script
  ├── evaluation_metrics.py  # Evaluation and metrics
  ├── dataset_loader.py      # Sample and synthetic data loading
  ├── config.py              # Configuration dataclasses
  ├── test_adaptive_chunking.py # Unit tests
  ├── .env                   # Environment variables (e.g., OpenAI API key)
  └── README.md              # This file
```

## Setup

1. **Clone the repository**
2. **Install dependencies** (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your OpenAI API key** in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

Run the full demo pipeline:
```bash
python -m enterprise-chunker.main_demo
```
This will:
- Initialize the system
- Load and process sample documents
- Apply adaptive chunking and store in ChromaDB
- Evaluate retrieval performance
- Run an interactive search demo
- Export results and evaluation reports

## Evaluation

- **Metrics**: Precision, Recall, F1 Score
- **Reports**: HTML and JSON reports are generated after each run
- **Sample Output:**
  - Precision: 66.67%
  - Recall: 66.67%
  - F1 Score: 66.67%

## Submission

- **GitHub Repository**: Complete implementation with evaluation metrics and demo scripts.

## License

MIT License (or specify your license here)

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://www.trychroma.com/)
- OpenAI 