import os
import logging
from typing import List, Optional
from pathlib import Path
import hashlib
import pickle

from unsloth import FastLanguageModel
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Fixed import
from langchain.prompts import PromptTemplate
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for caching
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
llm_cache = None
embedding_model_cache = None

class RAGChatbot:
    def __init__(self):
        self.llm = None
        self.embedding_model = None
        self.vector_store = None
        self.current_file_hash = None
        
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to check if it's changed"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_llm(self):
        """Get LLM with caching - optimized for Kaggle"""
        global llm_cache
        if llm_cache is None:
            logger.info("Loading LLM...")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                    device_map="auto",  # Auto device mapping
                    # Removed llm_int8_enable_fp32_cpu_offload for compatibility
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
                    device_map="auto",  # Auto device mapping for pipeline
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

    def load_documents(self, file_path: str) -> List:
        """Load documents from PDF with error handling"""
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages")
            return documents
        except Exception as e:
            error_msg = str(e).lower()
            if "encrypted" in error_msg or "password" in error_msg or "decrypted" in error_msg:
                logger.error(f"PDF is encrypted: {e}")
                raise Exception("❌ PDF is encrypted/password protected. Please remove the password protection and try again.")
            elif "not a pdf" in error_msg:
                logger.error(f"Invalid PDF file: {e}")
                raise Exception("❌ Invalid PDF file. Please upload a valid PDF document.")
            else:
                logger.error(f"Error loading PDF: {e}")
                raise Exception(f"❌ Failed to load PDF: {str(e)}")

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks with improved parameters"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,  # Increased overlap for better context
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Better separators
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

    def create_vector_store(self, chunks: List, file_hash: str):
        """Create vector store with caching - simplified for Kaggle"""
        cache_file = CACHE_DIR / f"vector_store_{file_hash}.pkl"
        
        # For Kaggle, skip caching to avoid pickle issues
        try:
            logger.info("Creating new vector store...")
            embeddings = self.get_embedding_model()
            self.vector_store = Chroma.from_documents(chunks, embeddings)
            logger.info("Vector store created successfully")
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def setup_rag_pipeline(self, file_path: str):
        """Setup the complete RAG pipeline"""
        file_hash = self.get_file_hash(file_path)
        
        # Check if we need to reload
        if self.current_file_hash != file_hash:
            logger.info("New file detected, reloading pipeline...")
            
            # Load and process documents
            documents = self.load_documents(file_path)
            chunks = self.split_documents(documents)
            
            # Create vector store
            self.create_vector_store(chunks, file_hash)
            
            self.current_file_hash = file_hash
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
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain

    def query_document(self, file_path: str, query: str) -> str:
        """Main function to query the document"""
        try:
            # Validate inputs
            if not file_path or not os.path.exists(file_path):
                return "❌ Error: Please upload a valid PDF file"
            
            if not query or query.strip() == "":
                return "❌ Error: Please enter a question"
            
            # Setup RAG pipeline
            self.setup_rag_pipeline(file_path)
            
            # Create QA chain
            qa_chain = self.create_qa_chain()
            
            # Get response
            logger.info(f"Processing query: {query}")
            result = qa_chain({"query": query})
            
            response = result["result"]
            
            # Add source information if available
            if result.get("source_documents"):
                sources = [f"Page {doc.metadata.get('page', 'Unknown')}" 
                          for doc in result["source_documents"]]
                response += f"\n\n📄 Sources: {', '.join(set(sources))}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in query_document: {e}")
            return f"❌ Error: {str(e)}"

# Initialize chatbot
chatbot = RAGChatbot()

# Gradio interface optimized for Kaggle
def create_interface():
    with gr.Blocks(
        title="🧠 RAG Chatbot with Mistral 7B",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🧠 RAG Chatbot with Mistral 7B
        
        Upload a PDF document and ask questions about its content. The system uses advanced AI to provide accurate, context-aware answers.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📄 Upload PDF File",
                    file_count="single",
                    file_types=['.pdf'],
                    type="filepath"
                )
                
                gr.Markdown("""
                ### 💡 Tips:
                - Upload any PDF document
                - Ask specific questions about the content
                - The system will search through the document and provide relevant answers
                """)
                
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="❓ Your Question",
                    lines=3,
                    placeholder="Ask a question about the PDF content...",
                    max_lines=5
                )
                
                submit_btn = gr.Button("🚀 Ask Question", variant="primary", size="lg")
                
                output = gr.Textbox(
                    label="🤖 Answer",
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
                
                # Example questions
                gr.Markdown("""
                ### 💭 Example Questions:
                - "What is the main topic of this document?"
                - "Summarize the key points"
                - "What are the conclusions?"
                - "Explain the methodology used"
                """)
        
        # Event handlers
        def process_query(file_path, query):
            if not file_path:
                return "❌ Please upload a PDF file first"
            return chatbot.query_document(file_path, query)
        
        submit_btn.click(
            fn=process_query,
            inputs=[file_input, query_input],
            outputs=output
        )
        
        # Allow Enter key to submit
        query_input.submit(
            fn=process_query,
            inputs=[file_input, query_input],
            outputs=output
        )
        
        # Clear button
        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        def clear_all():
            return None, "", ""
        
        clear_btn.click(
            fn=clear_all,
            outputs=[file_input, query_input, output]
        )
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,  # Enable sharing for Kaggle
        show_error=True
    ) 