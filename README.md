# 🧠 Advanced RAG Chatbot with Mistral 7B

A powerful local Retrieval-Augmented Generation (RAG) chatbot system that supports multiple input sources. Upload files, provide URLs, or type text directly to get intelligent, context-aware answers powered by Unsloth-optimized Mistral 7B.

## ✨ Features

- **🔒 Local Processing**: Runs completely offline, no internet dependency
- **📄 Multi-format Support**: PDF, TXT, MD, CSV files
- **🌐 URL Processing**: Extract content from web pages
- **✍️ Manual Text Input**: Type or paste any text content
- **🔄 Source Combination**: Combine multiple input sources
- **⚡ Optimized Model**: Unsloth-optimized Mistral 7B with 4-bit quantization
- **🔍 Semantic Search**: Advanced vector-based retrieval using ChromaDB
- **🎨 Modern UI**: Beautiful Gradio interface with intuitive design
- **💾 Memory Efficient**: Optimized for 8GB RAM systems
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
pip install -r requirements_advanced.txt

# Run the application
python app_advanced.py
```

### Usage

1. **Upload Files**: Click "Upload File" for PDF, TXT, MD, or CSV files
2. **Add URLs**: Paste web page URLs to extract content
3. **Type Text**: Enter or paste any text content directly
4. **Ask Questions**: Type your question and get intelligent answers
5. **Combine Sources**: Use multiple input types simultaneously

## 🎯 Supported Input Types

### 📄 File Upload
- **PDF**: Research papers, reports, manuals
- **TXT**: Plain text files
- **MD**: Markdown documents  
- **CSV**: Data files and tables

### 🌐 URL Input
- News articles and blog posts
- Documentation pages
- Wikipedia articles
- Technical guides
- Most public websites

### ✍️ Manual Text Input
- Notes and summaries
- Code snippets
- Meeting transcripts
- Personal research
- Any text content

## 🏗️ Architecture

### Core Components

1. **Language Model**: Unsloth Mistral 7B (4-bit quantized)
2. **Document Processing**: Multi-format loaders (PDF, Text, Web)
3. **Vector Database**: ChromaDB for semantic search
4. **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
5. **Web Interface**: Gradio with modern UI

### Workflow

```
Multiple Inputs → Content Extraction → Text Chunking → Embedding → Vector Store → Query → Retrieval → LLM Generation → Response
```

## 📦 Dependencies

### Core Libraries
- **LangChain**: RAG pipeline framework
- **Transformers**: Hugging Face transformers
- **Unsloth**: Model optimization
- **ChromaDB**: Vector database
- **Gradio**: Web interface

### Document Processing
- **PyPDF**: PDF reading and processing
- **Requests**: Web scraping
- **BeautifulSoup**: HTML parsing
- **Unstructured**: Advanced document processing

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

## 💡 Use Cases & Examples

### Academic Research
```
📄 Upload: Research paper PDF
🔗 URL: Related Wikipedia article
✍️ Text: Your analysis notes
❓ Question: "What are the main findings and how do they compare to existing literature?"
```

### Business Analysis
```
📄 Upload: Company report CSV
🔗 URL: Industry news article
✍️ Text: Market observations
❓ Question: "What are the key trends and opportunities identified?"
```

### Technical Documentation
```
📄 Upload: API documentation PDF
🔗 URL: GitHub repository README
✍️ Text: Implementation notes
❓ Question: "How do I implement authentication using this API?"
```

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

#### 2. **URL Loading Issues**
```
❌ Failed to load URL
```
**Solution**: 
- Check if URL is accessible
- Try a different URL
- Ensure website allows scraping

#### 3. **Out of Memory**
```
❌ CUDA out of memory
```
**Solution**: 
- Reduce `max_seq_length` in code
- Use CPU-only mode
- Increase system RAM

### Performance Tips

- **GPU Usage**: Enable GPU for faster inference
- **RAM Optimization**: Close other applications
- **File Size**: Keep files under 50MB
- **Text Quality**: Use text-based files, not scans

## 📁 Project Structure

```
Mistral_reader/
├── Backend/
│   ├── app_advanced.py           # Main application
│   ├── requirements_advanced.txt  # Dependencies
│   └── pdf_checker.py           # PDF validation tool
├── cache/                       # Model and vector cache
├── README.md                    # This file
├── PDF_Troubleshooting.md       # PDF issue guide
└── Advanced_Features_Guide.md   # Detailed features guide
```

## 🔄 Updates & Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements_advanced.txt
```

### Clear Cache
```bash
# Delete cache directory manually
rm -rf cache/
```

## 🌐 Deployment Options

### Local Development
```bash
python app_advanced.py
# Access at http://localhost:7860
```

### Docker (Coming Soon)
```bash
docker build -t advanced-rag-chatbot .
docker run -p 7860:7860 advanced-rag-chatbot
```

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Model Size | ~4GB (4-bit quantized) |
| Memory Usage | ~8GB RAM |
| Response Time | 2-5 seconds |
| Max File Size | 50MB |
| Supported Pages | Up to 500 pages |
| URL Processing | Most public websites |

## 🔒 Security & Privacy

- **Local Processing**: All data stays on your machine
- **No Internet**: Works completely offline
- **No Data Collection**: No telemetry or logging
- **Open Source**: Transparent codebase

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
pip install -r requirements_advanced.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint code
flake8
```

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
2. **Features Guide**: See `Advanced_Features_Guide.md`
3. **Troubleshooting**: See `PDF_Troubleshooting.md`
4. **Issues**: Create an issue on GitHub

### Common Questions

**Q: Why is my PDF not loading?**
A: Check if it's encrypted or password-protected. Use `pdf_checker.py` to diagnose.

**Q: Can I use this with URLs?**
A: Yes! Paste any public URL and the system will extract and process the content.

**Q: How can I combine multiple sources?**
A: Upload files, add URLs, and type text - the system processes all sources together.

**Q: What file types are supported?**
A: PDF, TXT, MD, and CSV files are supported.

**Q: How do I optimize for my hardware?**
A: Adjust `max_seq_length` and use GPU if available. See performance tips above.

---

**Made with ❤️ for the AI community**

*Built with Unsloth, LangChain, and Gradio* 