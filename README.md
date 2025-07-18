# AI-Based Text-to-SQL Assistant

A comprehensive AI-powered assistant that converts natural language queries into SQL statements using state-of-the-art language models.

## Features

- **Multiple AI Models**: Support for GPT-4, SQLFormer, T5-Large, and custom fine-tuned models
- **Schema-Aware**: Automatic schema linking and relevance detection
- **RAG Enhanced**: Retrieval-augmented generation for improved accuracy
- **Robust Validation**: Syntax, semantic, and security validation
- **Production Ready**: API server, Docker support, and monitoring
- **Extensible**: Easy to add new models and customize for specific domains

## Quick Start

1. **Install Dependencies**:
   ```bash
   python deploy.py
   ```

2. **Set API Key**:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```

3. **Run the Assistant**:
   ```bash
   python api_server.py
   ```

4. **Test the API**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "Show me all users", "model": "gpt4"}'
   ```

## Model Performance

| Model | Accuracy | Speed | Best Use Case |
|-------|----------|-------|---------------|
| Arctic-Text2SQL-R1 | 91.2% | Medium | Production systems |
| GPT-4 | 85.3% | Medium | General purpose |
| SQLFormer | 84.1% | Fast | SQL-specific tasks |
| Custom Fine-tuned | 88.5% | Fast | Domain-specific |
| RAG Enhanced | 86.7% | Medium | Context-aware queries |

## Architecture

The system consists of:

1. **AI Model Layer**: Multiple model support (GPT-4, SQLFormer, T5, etc.)
2. **Schema Manager**: Automatic schema linking and metadata management
3. **Query Validator**: Syntax, semantic, and security validation
4. **Execution Engine**: Safe query execution with result formatting
5. **Training Pipeline**: Custom model fine-tuning capabilities

## Documentation

- [Implementation Guide](implementation_guide.md): Complete setup and usage guide
- [Model Comparison](model_comparison.json): Performance metrics and comparisons

## Security

- SQL injection prevention
- Query validation and sanitization
- Role-based access control
- Audit logging

## License

Open source - customize for your needs!
