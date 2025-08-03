# 🧠 Local RAG Chatbot with Mistral 7B

A powerful local Retrieval-Augmented Generation (RAG) chatbot system using Unsloth-optimized Mistral 7B for intelligent document Q&A. Upload PDF documents and get contextually accurate answers powered by advanced AI.

## ✨ Features

- **🔒 Local Processing**: Runs completely offline, no internet dependency
- **📄 PDF Support**: Upload and process any PDF document
- **⚡ Optimized Model**: Unsloth-optimized Mistral 7B with 4-bit quantization
- **🔍 Semantic Search**: Advanced vector-based retrieval using ChromaDB
- **🎨 Modern UI**: Beautiful Gradio interface with intuitive design
- **💾 Memory Efficient**: Optimized for 8GB RAM systems
- **🔄 Smart Caching**: Intelligent caching for faster subsequent queries
- **📊 Source Tracking**: Shows page sources for all answers

## 🚀 Quick Start

### Prerequisites
- Python 3.8-3.11 (recommended 3.10)
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended (NVIDIA with 6GB+ VRAM)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Mistral_reader

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
cd Backend
pip install -r requirements.txt

# Run the application
python app.py
```

### Usage

1. **Upload PDF**: Click "Upload PDF File" and select your document
2. **Ask Questions**: Type your question in the text box
3. **Get Answers**: Receive contextually relevant answers with source pages

## 🏗️ Architecture

### Core Components

1. **Language Model**: Unsloth Mistral 7B (4-bit quantized)
2. **Document Processing**: PyPDFLoader + RecursiveCharacterTextSplitter
3. **Vector Database**: ChromaDB for semantic search
4. **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
5. **Web Interface**: Gradio with modern UI

### Workflow

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store → Query → Retrieval → LLM Generation → Response
```

## 📦 Dependencies

### Core Libraries
- **LangChain**: RAG pipeline framework
- **Transformers**: Hugging Face transformers
- **Unsloth**: Model optimization
- **ChromaDB**: Vector database
- **Gradio**: Web interface

### Models
- **Mistral 7B**: Base language model
- **Sentence Transformers**: Text embeddings

## 🔧 Configuration

### Model Parameters
- **Max Sequence Length**: 2048 tokens
- **Max New Tokens**: 512
- **Temperature**: 0.5
- **Top-k**: 50, Top-p**: 0.95
- **Repetition Penalty**: 1.1

### Text Processing
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Retrieval**: Top 4 most relevant chunks

## 🎯 Use Cases

### Academic Research
- Research paper analysis
- Literature review assistance
- Citation and reference tracking

### Business Applications
- Document Q&A
- Contract analysis
- Report summarization

### Educational Support
- Textbook comprehension
- Study material analysis
- Assignment assistance

### Technical Documentation
- Manual and guide queries
- API documentation search
- Troubleshooting assistance

## 🚨 Troubleshooting

### Common Issues

#### 1. **PDF Encrypted Error**
```
❌ PDF is encrypted/password protected
```
**Solution**: Remove password protection using:
- Adobe Acrobat
- Online tools (SmallPDF, ILovePDF)
- Print to PDF method

#### 2. **Out of Memory**
```
❌ CUDA out of memory
```
**Solution**: 
- Reduce `max_seq_length` in code
- Use CPU-only mode
- Increase system RAM

#### 3. **Model Download Slow**
```
❌ Slow model download
```
**Solution**:
- Use VPN or closer mirror
- Set `HF_ENDPOINT=https://hf-mirror.com`

### Performance Tips

- **GPU Usage**: Enable GPU for faster inference
- **RAM Optimization**: Close other applications
- **File Size**: Keep PDFs under 50MB
- **Text Quality**: Use text-based PDFs, not scans

## 📁 Project Structure

```
Mistral_reader/
├── Backend/
│   ├── app.py              # Main application
│   ├── app_kaggle.py       # Kaggle-optimized version
│   ├── requirements.txt     # Dependencies
│   ├── requirements_kaggle.txt
│   └── pdf_checker.py      # PDF validation tool
├── cache/                  # Model and vector cache
├── README.md              # This file
├── PDF_Troubleshooting.md # PDF issue guide
└── setup_environment.md   # Setup instructions
```

## 🔄 Updates & Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Clear Cache
```bash
python clear_models.py
```

### Check Disk Space
```bash
python check_disk_space.py
```

## 🌐 Deployment Options

### Local Development
```bash
python app.py
# Access at http://localhost:7860
```

### Kaggle Notebook
```bash
# Upload app_kaggle.py and requirements_kaggle.txt
!pip install -r requirements_kaggle.txt
!python app_kaggle.py
```

### Docker (Coming Soon)
```bash
docker build -t rag-chatbot .
docker run -p 7860:7860 rag-chatbot
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint code
flake8
```

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Model Size | ~4GB (4-bit quantized) |
| Memory Usage | ~8GB RAM |
| Response Time | 2-5 seconds |
| Max PDF Size | 50MB |
| Supported Pages | Up to 500 pages |

## 🔒 Security & Privacy

- **Local Processing**: All data stays on your machine
- **No Internet**: Works completely offline
- **No Data Collection**: No telemetry or logging
- **Open Source**: Transparent codebase

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Unsloth**: For model optimization
- **Hugging Face**: For transformers and models
- **LangChain**: For RAG framework
- **Gradio**: For web interface

## 📞 Support

### Getting Help

1. **Check Documentation**: Read this README thoroughly
2. **Troubleshooting Guide**: See `PDF_Troubleshooting.md`
3. **Issues**: Create an issue on GitHub
4. **Discussions**: Use GitHub Discussions

### Common Questions

**Q: Why is my PDF not loading?**
A: Check if it's encrypted or password-protected. Use `pdf_checker.py` to diagnose.

**Q: How can I improve response quality?**
A: Ask specific questions, use text-based PDFs, and ensure good document quality.

**Q: Can I use this with other file types?**
A: Currently supports PDF only. Future versions may support DOCX, TXT, etc.

**Q: How do I optimize for my hardware?**
A: Adjust `max_seq_length` and use GPU if available. See performance tips above.

---

**Made with ❤️ for the AI community**

*Built with Unsloth, LangChain, and Gradio* 