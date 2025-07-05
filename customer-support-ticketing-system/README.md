# Smart Customer Support Ticketing System

A comprehensive AI-powered customer support system that automatically categorizes incoming tickets and generates intelligent responses using RAG (Retrieval-Augmented Generation) architecture. The system analyzes historical tickets and company knowledge base to provide contextually relevant solutions.

## 🚀 Features

### Core Functionality
- **Smart Ticket Categorization**: AI-powered automatic categorization of support tickets
- **RAG-powered Responses**: Retrieval-Augmented Generation for intelligent response suggestions
- **Vector Search**: Semantic search using ChromaDB for knowledge base queries
- **Automated Priority Assignment**: Intelligent priority determination based on content analysis
- **Sentiment Analysis**: Customer sentiment detection for escalation decisions
- **Confidence Scoring**: Response confidence evaluation for human escalation

### Advanced Features
- **Multi-level Categorization**: Automatic tagging with confidence scoring
- **Solution Learning**: Learning from successful resolutions for future responses
- **Customer History Integration**: Personalized responses based on customer history
- **Escalation Logic**: Smart escalation triggers for complex issues
- **Real-time Processing**: Instant ticket processing and response generation

### Sample Use Cases (E-commerce)
- **"My order hasn't arrived"** → Auto-categorize as "Shipping Issue"
- **"I want to return this product"** → Generate response with return policy
- **"Payment failed but money deducted"** → Check similar resolved tickets and respond
- **"Product damaged during delivery"** → Auto-tag with refund process

## 🏗️ Architecture

### Backend Architecture
```
backend/
├── app/
│   ├── api/           # FastAPI routes and endpoints
│   ├── database/      # Database models and connections
│   ├── models/        # Pydantic models and SQLAlchemy schemas
│   ├── services/      # Business logic and RAG pipeline
│   ├── utils/         # Utility functions and text processing
│   └── main.py        # FastAPI application entry point
├── config.py          # Configuration settings
├── run.py             # Application entry point
└── pyproject.toml     # Project dependencies
```

### Frontend Architecture
```
frontend/
├── index.html         # Customer ticket submission interface
├── agent-dashboard.html  # Agent management dashboard
├── css/
│   └── style.css      # Styling and responsive design
└── js/
    ├── ticket-submission.js  # Customer interface logic
    └── agent-dashboard.js    # Agent dashboard logic
```

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM for database operations
- **PostgreSQL**: Primary database for ticket storage
- **ChromaDB**: Vector database for embeddings and semantic search
- **LangChain**: Framework for LLM applications and RAG pipeline
- **OpenAI**: LLM integration for response generation
- **Sentence Transformers**: Text embeddings for semantic search
- **Uvicorn**: ASGI server for production deployment

### Frontend
- **HTML5**: Semantic markup for accessibility
- **CSS3**: Modern styling with responsive design
- **JavaScript (ES6+)**: Interactive functionality and API integration
- **Fetch API**: Modern HTTP client for backend communication

## 📋 Prerequisites

Before running this application, ensure you have:

- **Python 3.11+** installed
- **PostgreSQL** database server running
- **UV package manager** (recommended) or pip
- **OpenAI API key** for LLM integration
- **Git** for version control

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd customer-support-ticketing-system
```

### 2. Backend Setup

#### Navigate to Backend Directory
```bash
cd backend
```

#### Install Dependencies
```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

#### Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

Required environment variables:
```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=support_system
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Application Security
SECRET_KEY=your_secret_key
```

#### Database Setup
```bash
# Create PostgreSQL database
createdb support_system

# Initialize database tables
python init_db.py
```

#### Start the Backend Server
```bash
python run.py
```

The API will be available at `http://localhost:8000`

### 3. Frontend Setup

#### Navigate to Frontend Directory
```bash
cd ../frontend
```

#### Serve the Frontend
```bash
# Using Python's built-in server
python -m http.server 8080

# Or using any static file server
npx serve .
```

The frontend will be available at `http://localhost:8080`

## 📖 API Documentation

### Available Endpoints

#### Tickets
- `GET /api/tickets` - List all tickets with optional filters
- `POST /api/tickets` - Create a new support ticket
- `GET /api/tickets/{ticket_id}` - Get specific ticket details
- `PUT /api/tickets/{ticket_id}/resolve` - Resolve ticket with solution
- `POST /api/tickets/{ticket_id}/response` - Generate smart response

#### Health Check
- `GET /health` - Application health status

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🎯 Usage Guide

### For Customers

1. **Submit a Ticket**
   - Navigate to the ticket submission page
   - Fill in your details (name, email, subject, description)
   - Submit the form
   - Receive instant categorization and suggested response

2. **View Ticket Status**
   - Your ticket will be automatically categorized
   - Priority will be assigned based on content
   - You'll receive a suggested response with confidence score

### For Support Agents

1. **Access Agent Dashboard**
   - Navigate to the agent dashboard
   - View all tickets with filtering options

2. **Manage Tickets**
   - Filter tickets by status, category, or priority
   - Click on any ticket to view details
   - Generate smart responses using RAG pipeline
   - Resolve tickets with custom solutions

3. **Smart Response Generation**
   - Click "Generate Smart Response" for AI-powered suggestions
   - Review confidence scores and sources
   - Use or modify suggested responses

## 🔧 RAG Pipeline Architecture

### 1. Ticket Ingestion
```python
# Text preprocessing and cleaning
text_processor = TextProcessor()
cleaned_text = text_processor.clean_text(ticket_content)
```

### 2. Categorization
```python
# Automatic categorization with confidence scoring
categorization_service = TicketCategorizationService()
category, confidence = categorization_service.categorize_ticket(subject, description)
```

### 3. Vector Storage
```python
# Store ticket embeddings in ChromaDB
chroma_service = ChromaDBService()
chroma_service.add_ticket(ticket_id, content, metadata)
```

### 4. Semantic Search
```python
# Search for similar tickets and knowledge base items
similar_tickets = chroma_service.search_similar_tickets(query, n_results=3)
kb_results = chroma_service.search_knowledge_base(query, n_results=2)
```

### 5. Response Generation
```python
# Generate contextual responses using RAG
rag_service = RAGService(openai_api_key)
response_data = rag_service.generate_response(
    ticket_content=query,
    similar_tickets=similar_ticket_data,
    knowledge_items=kb_items
)
```

## 📊 Database Schema

### Tickets Table
```sql
CREATE TABLE tickets (
    id SERIAL PRIMARY KEY,
    ticket_number VARCHAR(50) UNIQUE,
    customer_email VARCHAR(255),
    customer_name VARCHAR(255),
    subject VARCHAR(500),
    description TEXT,
    category VARCHAR(100),
    priority VARCHAR(20),
    status VARCHAR(50),
    sentiment_score FLOAT,
    confidence_score FLOAT,
    created_at TIMESTAMP,
    resolved_at TIMESTAMP,
    assigned_agent VARCHAR(255),
    resolution TEXT,
    customer_satisfaction INTEGER
);
```

### Knowledge Base Table
```sql
CREATE TABLE knowledge_base (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500),
    content TEXT,
    category VARCHAR(100),
    tags TEXT[],
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    is_active BOOLEAN
);
```

## 🎨 Frontend Features

### Customer Interface
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Feedback**: Loading states and success/error messages
- **Ticket Tracking**: View ticket details and suggested responses
- **Confidence Indicators**: Visual confidence scoring for responses

### Agent Dashboard
- **Advanced Filtering**: Filter by status, category, and priority
- **Smart Response Generation**: AI-powered response suggestions
- **Ticket Management**: Complete CRUD operations
- **Resolution Tracking**: Track ticket resolutions and agent assignments

## 🔒 Security Features

- **Input Validation**: Comprehensive input sanitization
- **SQL Injection Protection**: Parameterized queries with SQLAlchemy
- **CORS Configuration**: Proper cross-origin resource sharing
- **Environment Variables**: Secure configuration management
- **Error Handling**: Graceful error handling without information leakage

## 🧪 Testing

### Backend Testing
```bash
# Run backend tests
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
# Run frontend tests
cd frontend
npm test
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export DATABASE_URL=postgresql://user:pass@host:port/db
   ```

2. **Database Migration**
   ```bash
   python init_db.py
   ```

3. **Start Application**
   ```bash
   # Using Gunicorn for production
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t support-system .
docker run -p 8000:8000 support-system
```

## 📈 Performance Optimization

### Backend Optimizations
- **Connection Pooling**: Database connection optimization
- **Caching**: Redis integration for frequently accessed data
- **Async Processing**: Non-blocking I/O operations
- **Vector Indexing**: Optimized similarity search

### Frontend Optimizations
- **Lazy Loading**: Load components on demand
- **Caching**: Browser caching for static assets
- **Minification**: Compressed CSS and JavaScript
- **CDN Integration**: Content delivery network for assets

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests if applicable**
5. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful commit messages
- Add comprehensive documentation
- Include tests for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for RAG pipeline framework
- **ChromaDB** for vector database capabilities
- **FastAPI** for modern API development
- **OpenAI** for LLM integration
- **Sentence Transformers** for text embeddings

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation for common solutions

---

**Built with ❤️ for intelligent customer support automation** 