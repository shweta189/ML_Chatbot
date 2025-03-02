import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

HUGGINGFACE_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN')

# Step 1 : Setyp LLM (Mistral with HuggingFace)

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(repo_id = huggingface_repo_id,
                            temperature=0.5,
                            model_kwargs = {'token':HUGGINGFACE_API_TOKEN,
                                            'max_length':'512'})
    return llm

# step 2: Connect LLM with FAISS and Create chain
CUSTOMER_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
DB_FAISS_PATH = "ML_Chatbot\\vectorstore\\db_faiss"

def set_custom_prompt(custome_prompt_template):
    prompt = PromptTemplate(template=custome_prompt_template,input_variables=['context','question'])
    return prompt

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

# Create QA Chains

qa_chains = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type= "stuff",
    retriever= db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs= {'prompt':set_custom_prompt(CUSTOMER_PROMPT_TEMPLATE)}
)

# Invoking the chains

user_query = input("Write Query Here: ")
response = qa_chains.invoke({'query':user_query})

print("Result ",response['result'])
print("Source Documents ",response['source_documents'])
