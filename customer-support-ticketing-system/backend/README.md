# Customer Support Ticketing System - Backend

A smart customer support ticketing system with AI-powered categorization and RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Smart Ticket Categorization**: AI-powered automatic categorization of support tickets
- **RAG-powered Responses**: Retrieval-Augmented Generation for intelligent response suggestions
- **Vector Search**: Semantic search using ChromaDB for knowledge base queries
- **RESTful API**: FastAPI-based REST API for ticket management
- **PostgreSQL Database**: Reliable data storage with SQLAlchemy ORM

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Text embeddings for semantic search
- **LangChain**: Framework for LLM applications
- **PostgreSQL**: Primary database
- **Uvicorn**: ASGI server

## Project Structure

```
backend/
├── app/
│   ├── api/           # API routes
│   ├── database/      # Database models and connections
│   ├── models/        # Pydantic models
│   ├── services/      # Business logic
│   ├── utils/         # Utility functions
│   └── main.py        # FastAPI application
├── config.py          # Configuration settings
├── run.py             # Application entry point
├── .env               # Environment variables
└── pyproject.toml     # Project dependencies
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL
- UV package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customer-support-ticketing-system/backend
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Set up PostgreSQL database**
   ```bash
   # Create database
   createdb support_system
   ```

5. **Run the application**
   ```bash
   python run.py
   ```

## API Endpoints

### Tickets

- `GET /api/tickets` - List all tickets
- `POST /api/tickets` - Create a new ticket
- `GET /api/tickets/{ticket_id}` - Get ticket details
- `PUT /api/tickets/{ticket_id}` - Update ticket
- `DELETE /api/tickets/{ticket_id}` - Delete ticket

### Health Check

- `GET /health` - Application health status

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_HOST` | PostgreSQL host | localhost |
| `POSTGRES_PORT` | PostgreSQL port | 5432 |
| `POSTGRES_DB` | Database name | support_system |
| `POSTGRES_USER` | Database user | postgres |
| `POSTGRES_PASSWORD` | Database password | your_password |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | ./chroma_db |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `SECRET_KEY` | JWT secret key | your_secret_key |

## Development

### Running in Development Mode

```bash
python run.py
```

The API will be available at `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.
