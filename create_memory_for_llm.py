from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
# step1 : Load raw PDF(s)

DATA_PATH = "ML_Chatbot\data"

def load_pdf_files(data):
    loader = DirectoryLoader(data,
                        glob='*.pdf',
                        loader_cls=PyPDFLoader)
    documents= loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
# print('length of PDF pages: ',len(documents))

# Step 2: create chunks 
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)


text_chunks = create_chunks(extracted_data=documents)

# print("Length of Chunks: ",len(text_chunks))
# Step 3 : Vector Embedding

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4 : Store Embedding in Faiss
DB_FAISS_PATH= "ML_Chatbot\\vectorstore\\db_faiss"
db = FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)


# Step 5 : User Interface
