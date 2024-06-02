import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

OpenAIModel = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=OpenAIModel, temperature=0.1, openai_api_key=OPENAI_API_KEY)

def ask(text):
    answer = qa.run(text)
    return answer

loader = TextLoader("NutriGrade dataset.txt")
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)
db2 = FAISS.from_documents(all_splits, embeddings)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db2.as_retriever())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class Item(BaseModel):
    message: str

@app.post("/api/generate")
def generate_text(item: Item):
    outputs = ask(item.message)
    return {"answer": outputs}
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

OpenAIModel = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=OpenAIModel, temperature=0.1, openai_api_key=OPENAI_API_KEY)

def ask(text):
    answer = qa.run(text)
    return answer

loader = TextLoader("NutriGrade dataset.txt")
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)
db2 = FAISS.from_documents(all_splits, embeddings)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db2.as_retriever())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class Item(BaseModel):
    message: str

@app.post("/api/generate")
def generate_text(item: Item):
    outputs = ask(item.message)
    return {"answer": outputs}
