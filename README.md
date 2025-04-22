# Confluence Knowledge Agent

A production-ready API service that connects to Confluence, indexes content, and provides AI-powered answers to user queries.

## Features

- **Secure Environment Configuration**: All sensitive information stored in environment variables
- **FastAPI REST API**: Production-grade API with proper error handling and validation
- **Vector Search**: Uses Milvus for efficient hybrid search of Confluence content
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Background Processing**: Handles long-running tasks in the background
- **Health Checks**: Built-in health monitoring
- **Logging**: Comprehensive logging for monitoring and debugging

## Setup

1. Clone this repository
2. Configure your environment variables in the `.env` file
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `python -m code.main`

## Using Docker

Build and start the service:

```bash
docker-compose up -d
```

## API Endpoints

### Query the Knowledge Base

```
POST /api/query
```

Request body:
```json
{
  "query": "What is the procedure for onboarding a new customer?",
  "chat_history": [] // Optional
}
```

Response:
```json
{
  "answer": "The procedure for onboarding a new customer involves...",
  "response_time": 1.25,
  "sources": [
    {
      "content": "...",
      "source": "https://confluence.example.com/page",
      "title": "Customer Onboarding Procedure",
      "score": 0.92
    }
  ]
}
```

### Load Confluence Data

```
POST /api/load-confluence
```

Request body:
```json
{
  "force_reload": false
}
```

Response:
```json
{
  "status": "started",
  "message": "Confluence data loading process has been started in the background"
}
```

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Milvus connection
MILVUS_URI=
MILVUS_TOKEN=
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_DB_NAME=

# OpenAI API
OPENAI_API_KEY=
OPENAI_MODEL_NAME=
EMBEDDING_MODEL_NAME=
EMBEDDING_DIMENSIONS=

# Confluence connection
CONFLUENCE_URL=
CONFLUENCE_USERNAME=
CONFLUENCE_TOKEN=
CONFLUENCE_SPACE_KEY=
CONFLUENCE_INCLUDE_ATTACHMENTS=
CONFLUENCE_LIMIT=
CONFLUENCE_OCR_LANGUAGES=
CONFLUENCE_MAX_PAGES=

# API Settings
API_HOST=
API_PORT=
API_WORKERS=
```

## Security Considerations

- The `.env` file contains sensitive information and should not be committed to version control
- In production, restrict CORS settings to only allow specific origins
- Use a reverse proxy like Nginx for additional security
- Consider implementing rate limiting and authentication for the API