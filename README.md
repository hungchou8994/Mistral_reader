# 🧠 Local RAG Chatbot with Mistral 7B

A local RAG (Retrieval-Augmented Generation) chatbot using Unsloth Mistral 7B to answer questions based on uploaded PDF content.

## ✨ Features

- **Local Processing**: Runs completely on local machine, no internet connection required
- **PDF Support**: Upload and process PDF files
- **Optimized Model**: Uses Unsloth to optimize Mistral 7B with 4-bit quantization
- **Vector Search**: Uses ChromaDB for vector storage and retrieval
- **Modern UI**: Beautiful web interface with Gradio
- **Memory Efficient**: Only uses ~8GB RAM with 4-bit model

## 🚀 Installation

### System Requirements
- Python 3.8+
- RAM: Minimum 8GB (recommended 16GB)
- GPU: Not required but recommended for faster inference

### Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd Mistral_reader

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
cd Backend
pip install -r requirements.txt
```

## 🎯 Usage

### Run Application

```bash
cd Backend
python app.py
```

After successful startup, the application will open at: `http://localhost:7860`

### How to Use

1. **Upload PDF**: Click on "Upload PDF File" and select the PDF file to process
2. **Enter Query**: Type your question in the "Input Query" box
3. **Get Results**: The system will return an answer based on the PDF content

## 🏗️ Architecture

### Main Components

1. **Language Model**: Unsloth Mistral 7B (4-bit quantized)
2. **Document Loader**: PyPDFLoader to read PDF files
3. **Text Splitter**: RecursiveCharacterTextSplitter to chunk text
4. **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
5. **Vector Store**: ChromaDB for storage and retrieval
6. **Retrieval QA**: LangChain RetrievalQA chain
7. **UI**: Gradio interface

### Workflow

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store → Query → Retrieval → LLM Generation → Response
```

## 📦 Dependencies

### Core Libraries
- **LangChain**: Framework for RAG pipeline
- **Transformers**: Hugging Face transformers library
- **Unsloth**: Model inference optimization
- **ChromaDB**: Vector database
- **Gradio**: Web interface

### Model & Embeddings
- **Mistral 7B**: Base language model
- **Sentence Transformers**: Text embeddings

## 🔧 Configuration

### Model Parameters
- **Max Sequence Length**: 2048 tokens
- **Max New Tokens**: 512
- **Temperature**: 0.5
- **Top-k**: 50
- **Top-p**: 0.95
- **Repetition Penalty**: 1.1

### Text Processing
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 50 characters

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `max_seq_length` or use a smaller model
2. **PDF not readable**: Ensure the PDF file is not encrypted
3. **Slow model download**: Use VPN or closer mirror

### Performance Tips

- Use GPU for faster inference
- Increase RAM if processing large PDF files
- Adjust chunk size based on content

## 📝 License

This project uses open source libraries. Please check the license of each dependency.

## 🤝 Contributing

All contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

If you encounter any issues, please create an issue on the GitHub repository. 