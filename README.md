# 🚗 Recommendation Chatbot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, intelligent chatbot for Michelin product recommendations using Retrieval-Augmented Generation (RAG) with state-of-the-art AI technologies.

## 📱 Application Interface

The application features a modern, intuitive interface with:
- **🚗 Tire Assistant**: Clean, professional branding
- **💬 Interactive Chat**: Natural language tire recommendations
- **⚡ Quick Actions**: Pre-built examples for common queries
- **🎛️ System Controls**: Real-time status monitoring and chat management
- **🌐 Multi-language Support**: Available in French and English
- **📊 Source Documents**: Transparent recommendations with source references

### Key Interface Features:
- **Welcome Message**: Guides users on available tire categories and options
- **Smart Suggestions**: Quick action buttons for common tire needs
- **Real-time Status**: System health monitoring in the sidebar
- **Chat History**: Persistent conversation management
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## ✨ Features

- 🔍 **Semantic Search**: Advanced OpenAI embeddings for understanding user intent
- 🤖 **AI-Powered Responses**: GPT-4o-mini for natural, contextual recommendations
- 🌐 **Modern Web Interface**: Beautiful Streamlit UI with responsive design
- ⚡ **High-Performance Vector Search**: Local FAISS index for fast, private searches
- 🛠️ **CLI Support**: Command-line interface for testing and automation
- 🌍 **Multilingual**: Support for English and Italian
- 📊 **Comprehensive Logging**: Structured logging with rich formatting
- 🔒 **Type Safety**: Full type hints and Pydantic validation
- 🧪 **Extensive Testing**: Comprehensive test suite with pytest
- 📦 **Modern Python**: Built with Python 3.11+ and latest best practices

## 🏗️ Architecture

```
michelin-chatbot/
├── 📁 src/                          # Source code
│   ├── 📄 __init__.py               # Package initialization
│   ├── ⚙️ config.py                 # Configuration management
│   ├── 🚀 main.py                   # Main entry point
│   ├── 📁 models/                   # Data models
│   │   ├── 📄 __init__.py
│   │   └── 🏷️ tire.py               # Pydantic models
│   ├── 📁 services/                 # Business logic
│   │   ├── 📄 __init__.py
│   │   └── 🧠 rag_service.py        # RAG implementation
│   ├── 📁 ui/                       # User interfaces
│   │   ├── 📄 __init__.py
│   │   └── 🎨 streamlit_app.py      # Streamlit interface
│   └── 📁 utils/                    # Utilities
│       ├── 📄 __init__.py
│       └── 📝 logging.py            # Logging utilities
├── 📁 tests/                        # Test suite
│   ├── 📄 __init__.py
│   ├── 🧪 test_config.py
│   ├── 🧪 test_models.py
│   ├── 🧪 test_rag_service.py
│   └── 🧪 test_streamlit_app.py
├── 📄 catalog.csv                   # Product catalog
├── 📄 requirements.txt              # Dependencies
├── 📄 pyproject.toml                # Project configuration
├── 📄 .env.example                  # Environment template
├── 📄 .gitignore                    # Git ignore rules
├── 📄 reco_michelin_chatbot.py      # Legacy entry point
└── 📄 README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd michelin-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Activate (Linux/Mac)
   source .venv/bin/activate
   
   # Activate (Windows)
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your OpenAI API key
   OPENAI_API_KEY=sk-your-api-key-here
   ```

### Usage

#### 🌐 Web Interface (Recommended)
```bash
# Using the new modular interface
python -m src.main web

# Or using the legacy entry point
streamlit run reco_michelin_chatbot.py
```
Then open http://localhost:8501

#### 💻 Command Line Interface
```bash
# Interactive CLI mode
python -m src.main cli

# Check system status
python -m src.main status

# Rebuild search index
python -m src.main rebuild-index

# Show help
python -m src.main --help
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | ✅ |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` | ❌ |
| `CHAT_MODEL` | OpenAI chat model | `gpt-4o-mini` | ❌ |
| `TEMPERATURE` | Response creativity (0-1) | `0.2` | ❌ |
| `TOP_K` | Documents to retrieve | `4` | ❌ |
| `LANGUAGE` | Default language (en/it) | `en` | ❌ |
| `LOG_LEVEL` | Logging level | `INFO` | ❌ |
| `STREAMLIT_HOST` | Streamlit host | `localhost` | ❌ |
| `STREAMLIT_PORT` | Streamlit port | `8501` | ❌ |

### Catalog Format

The `catalog.csv` file should follow this structure:

```csv
id,name,description,category,price,link
PIL-PS5,Michelin Pilot Sport 5,High performance summer tire for sports cars,Summer,179.90,https://example.com/ps5
CRO-CROSS,Michelin CrossClimate 2,All-season tire with excellent snow traction,All-Season,149.50,https://example.com/crossclimate
```

## 🧪 Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Development Installation
```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt
```

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m src.main status || exit 1

# Run application
CMD ["python", "-m", "src.main", "web", "--host", "0.0.0.0"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  michelin-chatbot:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./catalog.csv:/app/catalog.csv
      - ./logs:/app/logs
```

## 🔧 Customization

### Adding New Languages
1. Update `src/config.py` to include the new language
2. Add translations in `src/ui/streamlit_app.py`
3. Update prompts in `src/services/rag_service.py`

### Custom Product Models
1. Modify `src/models/tire.py` to add new fields
2. Update the catalog CSV format
3. Adjust the RAG service accordingly

### UI Customization
1. Edit `src/ui/streamlit_app.py` for layout changes
2. Modify CSS in the `render_custom_css()` function
3. Add new components as needed

## 📊 Monitoring & Logging

The application includes comprehensive logging:

- **Console Output**: Rich formatted logs for development
- **File Logging**: Rotating log files in `logs/` directory
- **Structured Logging**: JSON format for production environments
- **Performance Metrics**: Query timing and system health

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format code (`black .`)
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation for new features
- Use conventional commit messages

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for the powerful language models
- [LangChain](https://langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the beautiful web interface
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [Pydantic](https://pydantic.dev/) for data validation

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](../../issues) page
2. Review the documentation
3. Create a new issue with detailed information

---

**Made with ❤️ for better tire recommendations**
