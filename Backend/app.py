from unsloth import FastLanguageModel
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import gradio as gr

# 1. Load Unsloth Mistral 7B (4bit)
def get_llm():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        max_seq_length=2048,
        dtype=None,  # auto
        load_in_4bit=True,
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
        return_full_text=True
    )

    return HuggingFacePipeline(pipeline=generation_pipe)

# 2. Load PDF
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# 3. Text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_documents(data)

# 4. Embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Vector store
def vector_database(chunks):
    embeddings = get_embedding_model()
    vectordb = Chroma.from_documents(chunks, embeddings)
    return vectordb

# 6. Retriever
def retriever(file_path):
    splits = document_loader(file_path)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

# 7. QA Chain
def retriever_qa(file_path, query):
    try:
        llm = get_llm()
        retriever_obj = retriever(file_path)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_obj, return_source_documents=False)
        response = qa.run(query)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 8. Gradio app
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="🧠 Local RAG Chatbot",
    description="Upload a PDF and ask questions. Powered by Unsloth Mistral 7B."
)

# 9. Launch app
rag_application.launch(server_name="0.0.0.0", server_port=7860)
