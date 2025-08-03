import os
import logging
import requests
from typing import List, Optional, Union
from pathlib import Path
import hashlib
import pickle
from urllib.parse import urlparse
import re

from unsloth import FastLanguageModel
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    WebBaseLoader,
    UnstructuredURLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for caching
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
llm_cache = None
embedding_model_cache = None

class AdvancedRAGChatbot:
    def __init__(self):
        self.llm = None
        self.embedding_model = None
        self.vector_store = None
        self.current_content_hash = None
        
    def get_content_hash(self, content: str) -> str:
        """Generate hash for content to check if it's changed"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_llm(self):
        """Get LLM with caching"""
        global llm_cache
        if llm_cache is None:
            logger.info("Loading LLM...")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                    device_map="auto",
                )

                generation_pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.5,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    return_full_text=True,
                    device_map="auto",
                )
                
                llm_cache = HuggingFacePipeline(pipeline=generation_pipe)
                logger.info("LLM loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LLM: {e}")
                raise
        return llm_cache

    def get_embedding_model(self):
        """Get embedding model with caching"""
        global embedding_model_cache
        if embedding_model_cache is None:
            logger.info("Loading embedding model...")
            try:
                embedding_model_cache = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise
        return embedding_model_cache

    def load_pdf_documents(self, file_path: str) -> List[Document]:
        """Load documents from PDF"""
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            error_msg = str(e).lower()
            if "encrypted" in error_msg or "password" in error_msg:
                raise Exception("❌ PDF is encrypted/password protected. Please remove the password protection.")
            else:
                raise Exception(f"❌ Failed to load PDF: {str(e)}")

    def load_text_documents(self, file_path: str) -> List[Document]:
        """Load documents from text file"""
        try:
            logger.info(f"Loading text file: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f"Loaded text file: {len(documents)} chunks")
            return documents
        except Exception as e:
            raise Exception(f"❌ Failed to load text file: {str(e)}")

    def load_url_content(self, url: str) -> List[Document]:
        """Load content from URL"""
        try:
            logger.info(f"Loading URL: {url}")
            
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise Exception("❌ Invalid URL format")
            
            # Try different loaders
            try:
                # Try WebBaseLoader first
                loader = WebBaseLoader(url)
                documents = loader.load()
            except:
                # Fallback to UnstructuredURLLoader
                loader = UnstructuredURLLoader(urls=[url])
                documents = loader.load()
            
            logger.info(f"Loaded URL content: {len(documents)} chunks")
            return documents
        except Exception as e:
            raise Exception(f"❌ Failed to load URL: {str(e)}")

    def create_documents_from_text(self, text: str, source: str = "Manual Input") -> List[Document]:
        """Create documents from manual text input"""
        try:
            logger.info(f"Creating documents from manual text (length: {len(text)})")
            
            # Split text into chunks
            chunk_size = 1000
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "chunk": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} documents from manual text")
            return documents
        except Exception as e:
            raise Exception(f"❌ Failed to process manual text: {str(e)}")

    def load_documents_from_input(self, file_path: str = None, url: str = None, manual_text: str = None) -> List[Document]:
        """Load documents from various input sources"""
        documents = []
        
        # Load from file
        if file_path and os.path.exists(file_path):
            if file_path.lower().endswith('.pdf'):
                documents.extend(self.load_pdf_documents(file_path))
            elif file_path.lower().endswith(('.txt', '.md', '.csv')):
                documents.extend(self.load_text_documents(file_path))
            else:
                raise Exception("❌ Unsupported file type. Please use PDF, TXT, MD, or CSV files.")
        
        # Load from URL
        if url and url.strip():
            documents.extend(self.load_url_content(url.strip()))
        
        # Load from manual text
        if manual_text and manual_text.strip():
            documents.extend(self.create_documents_from_text(manual_text.strip()))
        
        if not documents:
            raise Exception("❌ No valid input provided. Please provide a file, URL, or text.")
        
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with improved parameters"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

    def create_vector_store(self, chunks: List[Document], content_hash: str):
        """Create vector store with caching"""
        cache_file = CACHE_DIR / f"vector_store_{content_hash}.pkl"
        
        # For simplicity, skip caching to avoid pickle issues
        try:
            logger.info("Creating new vector store...")
            embeddings = self.get_embedding_model()
            self.vector_store = Chroma.from_documents(chunks, embeddings)
            logger.info("Vector store created successfully")
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def setup_rag_pipeline(self, file_path: str = None, url: str = None, manual_text: str = None):
        """Setup the complete RAG pipeline"""
        # Create content identifier
        content_parts = []
        if file_path:
            content_parts.append(f"file:{file_path}")
        if url:
            content_parts.append(f"url:{url}")
        if manual_text:
            content_parts.append(f"text:{self.get_content_hash(manual_text)}")
        
        content_id = "|".join(content_parts)
        content_hash = hashlib.md5(content_id.encode()).hexdigest()
        
        # Check if we need to reload
        if self.current_content_hash != content_hash:
            logger.info("New content detected, reloading pipeline...")
            
            # Load and process documents
            documents = self.load_documents_from_input(file_path, url, manual_text)
            chunks = self.split_documents(documents)
            
            # Create vector store
            self.create_vector_store(chunks, content_hash)
            
            self.current_content_hash = content_hash
            logger.info("RAG pipeline setup complete")

    def create_qa_chain(self):
        """Create QA chain with custom prompt"""
        custom_prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""

        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )

        llm = self.get_llm()
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain

    def query_content(self, file_path: str = None, url: str = None, manual_text: str = None, query: str = None) -> str:
        """Main function to query the content"""
        try:
            # Validate inputs
            if not query or query.strip() == "":
                return "❌ Error: Please enter a question"
            
            if not file_path and not url and not manual_text:
                return "❌ Error: Please provide a file, URL, or text input"
            
            # Setup RAG pipeline
            self.setup_rag_pipeline(file_path, url, manual_text)
            
            # Create QA chain
            qa_chain = self.create_qa_chain()
            
            # Get response
            logger.info(f"Processing query: {query}")
            result = qa_chain({"query": query})
            
            response = result["result"]
            
            # Add source information if available
            if result.get("source_documents"):
                sources = []
                for doc in result["source_documents"]:
                    source = doc.metadata.get('source', 'Unknown')
                    if 'page' in doc.metadata:
                        sources.append(f"{source} (Page {doc.metadata['page']})")
                    elif 'chunk' in doc.metadata:
                        sources.append(f"{source} (Chunk {doc.metadata['chunk']})")
                    else:
                        sources.append(source)
                
                response += f"\n\n📄 Sources: {', '.join(set(sources))}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in query_content: {e}")
            return f"❌ Error: {str(e)}"

# Initialize chatbot
chatbot = AdvancedRAGChatbot()

# Gradio interface for advanced RAG
def create_advanced_interface():
    with gr.Blocks(
        title="🧠 Advanced RAG Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🧠 Advanced RAG Chatbot with Multiple Input Sources
        
        Upload files, provide URLs, or type text directly. The system uses advanced AI to provide accurate, context-aware answers from any source.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 File Upload")
                file_input = gr.File(
                    label="📄 Upload File (PDF, TXT, MD, CSV)",
                    file_count="single",
                    file_types=['.pdf', '.txt', '.md', '.csv'],
                    type="filepath"
                )
                
                gr.Markdown("### 🌐 URL Input")
                url_input = gr.Textbox(
                    label="🔗 Enter URL",
                    placeholder="https://example.com/article",
                    lines=2
                )
                
                gr.Markdown("### ✍️ Manual Text Input")
                manual_text_input = gr.Textbox(
                    label="📝 Enter your text here",
                    placeholder="Paste or type your content here...",
                    lines=8,
                    max_lines=15
                )
                
                gr.Markdown("""
                ### 💡 Tips:
                - **Files**: Upload PDF, TXT, MD, or CSV files
                - **URLs**: Provide web page URLs to extract content
                - **Text**: Type or paste any text content
                - **Multiple Sources**: You can combine different input types
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("### ❓ Ask Questions")
                query_input = gr.Textbox(
                    label="❓ Your Question",
                    lines=3,
                    placeholder="Ask a question about the content...",
                    max_lines=5
                )
                
                submit_btn = gr.Button("🚀 Ask Question", variant="primary", size="lg")
                
                output = gr.Textbox(
                    label="🤖 Answer",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                
                # Example questions
                gr.Markdown("""
                ### 💭 Example Questions:
                - "What is the main topic?"
                - "Summarize the key points"
                - "What are the conclusions?"
                - "Explain the methodology"
                - "What are the main arguments?"
                """)
        
        # Event handlers
        def process_query(file_path, url, manual_text, query):
            if not query:
                return "❌ Please enter a question"
            
            if not file_path and not url and not manual_text:
                return "❌ Please provide at least one input source (file, URL, or text)"
            
            return chatbot.query_content(file_path, url, manual_text, query)
        
        submit_btn.click(
            fn=process_query,
            inputs=[file_input, url_input, manual_text_input, query_input],
            outputs=output
        )
        
        # Allow Enter key to submit
        query_input.submit(
            fn=process_query,
            inputs=[file_input, url_input, manual_text_input, query_input],
            outputs=output
        )
        
        # Clear button
        clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        def clear_all():
            return None, "", "", "", ""
        
        clear_btn.click(
            fn=clear_all,
            outputs=[file_input, url_input, manual_text_input, query_input, output]
        )
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    interface = create_advanced_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 