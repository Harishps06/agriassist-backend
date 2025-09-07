import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- NEW IMPORTS FOR THE AI BRAIN ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# --- SETUP THE AI BRAIN (This runs only once when the server starts) ---

# 1. Put your Google API Key here
os.environ["GOOGLE_API_KEY"] = "AIzaSyCWK3gI22NlZXOqNFSpj8ag3yR752uj6tU"

# 2. Load the documents from your knowledge_base folder (Handles both PDF and TXT)
# NOTE: Important security warning about your API Key is below.
pdf_loader = DirectoryLoader('./knowledge_base/', glob="**/*.pdf", loader_cls=PyPDFLoader)
txt_loader = DirectoryLoader('./knowledge_base/', glob="**/*.txt", loader_cls=TextLoader)
pdf_documents = pdf_loader.load()
txt_documents = txt_loader.load()
documents = pdf_documents + txt_documents


# 3. Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 4. Create embeddings (numerical representations of text)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Create a Vector Store to search for relevant chunks
#    This will create a 'db' folder for the vector store
vector_store = Chroma.from_documents(texts, embeddings, persist_directory="db")
vector_store.persist()
vector_store = None

# 6. Setup the LLM and the Retrieval QA Chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, convert_system_message_to_human=True)
# We will define the final qa_chain when we run the app
qa_chain = None 
# --- END OF AI SETUP ---


# Define an API endpoint for handling text queries
@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    print(f"Server received a question: {question}")

    # --- THIS IS THE FINAL, ACTIVATED AI LOGIC ---
    # It uses the qa_chain to get an answer from your documents.
    if qa_chain:
        response = qa_chain.invoke({"query": question})
        response_text = response['result']
    else:
        response_text = "AI chain is not ready. Please wait a moment and try again."
        
    return jsonify({'answer': response_text})

# Run the app
if __name__ == '__main__':
    
    # Reload the vector store from disk
    vector_store_from_disk = Chroma(persist_directory="db", embedding_function=embeddings)
    
    # Create a retriever to fetch relevant documents
    retriever = vector_store_from_disk.as_retriever(search_kwargs={"k": 2})

    # Create the prompt template
    prompt_template = """
    You are an expert agricultural assistant for farmers in Kerala, India. Use the following pieces of context to answer the user's question.
    If you don't know the answer from the context, just say that you don't have enough information, don't try to make up an answer.
    Answer in a clear, simple, and helpful way.

    CONTEXT: {context}

    QUESTION: {question}

    HELPFUL ANSWER:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create the final QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    app.run(debug=True, port=5000)
