from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel, ValidationError
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
import json
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    loader = PyPDFLoader('IITM BS Degree Programme - Student Handbook - Latest.pdf')
    documents = loader.load()
except Exception as e:
    logging.error(f"Error loading documents: {e}")
    documents = []
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


# Load embeddings
# embeddings = HuggingFaceEmbeddings()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts,embeddings)
# print(db)


retriever = db.as_retriever(search_kwargs={'k': 5})

# Initialize the LLM (Hugging Face Hub)
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceHub(
    huggingfacehub_api_token='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',
    repo_id=repo_id,
    model_kwargs={"temperature": 0.2, "max_new_tokens": 50}
)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever,return_source_documents=True)
@app.route("/query", methods=["POST"])
def get_answer():
    data = json.loads(request.data.decode('utf-8'))
    print(data)
    
    # Assuming 'query' is a key in the JSON data sent from the client
    query = data.get('query', '')
    
    if query:
        # Replace with your actual function to get the answer
        result = qa_chain({'question': query, 'chat_history': []})
        answer = result.get('answer', 'No answer found')
        marker = "Helpful Answer:"

        # Find the position of the marker
        start_pos = answer.find(marker)

        # Check if the marker is found
        if start_pos != -1:
            # Extract the text after the marker
            answer = answer[start_pos + len(marker):].strip()
            print(answer)
        else:
            print("Marker not found")
    else:
        answer = 'No query provided'
    
    # Return the answer as JSON
    return jsonify({'answer': answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)