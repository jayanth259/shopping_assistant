import qrcode
from pyngrok import ngrok
from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from py2neo import Graph
import requests

# Step 1: Convert Tabular Data to Graph DB
def load_data_to_graph(csv_path, graph_uri, user, password):
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    graph = Graph(graph_uri, auth=(user, password))
    for doc in docs:
        graph.run("""
            MERGE (p:Product {name: $name, category: $category, price: $price})
        """, parameters=doc.metadata)
    return graph

# Step 2: Convert data into embeddings
def create_faiss_vector_store(csv_path):
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Step 3: Initialize Llama Model
def initialize_llama_model():
    llm = LlamaCpp(model_path="llama-2-7b-chat.ggmlv3.q4_0.bin", temperature=0.7, max_tokens=256)
    return llm

# Step 4: Build Advanced RAG Pipeline
def setup_advanced_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever()
    llm = initialize_llama_model()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    return qa_chain

# Step 5: Flask App for WhatsApp Integration
app = Flask(__name__)
qa_chain = None

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = qa_chain.run({'question': user_message, 'chat_history': []})
    return jsonify({'reply': response})

# Step 6: Send Message via WhatsApp Business API
def send_whatsapp_message(phone_number, message):
    url = "https://graph.facebook.com/v15.0/your-whatsapp-number/messages"
    headers = {
        "Authorization": "Bearer YOUR_ACCESS_TOKEN",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Step 7: Generate QR Code for WhatsApp
def generate_qr_code():
    url = "https://wa.me/your-whatsapp-number"
    qr = qrcode.make(url)
    qr.save("whatsapp_qr.png")
    print("QR Code generated: whatsapp_qr.png")

if __name__ == "__main__":
    csv_file_path = "ecommerce_data.csv"
    graph_uri = "bolt://localhost:7687"
    graph_user = "neo4j"
    graph_password = "password"

    # Load data to Graph DB
    load_data_to_graph(csv_file_path, graph_uri, graph_user, graph_password)
    
    # Create FAISS store
    vector_store = create_faiss_vector_store(csv_file_path)
    
    # Setup Advanced RAG
    qa_chain = setup_advanced_rag_pipeline(vector_store)
    
    # Start Flask App
    public_url = ngrok.connect(5000)
    print(f"Chatbot running at: {public_url}")
    generate_qr_code()
    app.run(port=5000)
