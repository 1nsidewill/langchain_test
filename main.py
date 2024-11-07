from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, has_collection, drop_collection, Index, WeightedRanker
import os
import zipfile
import shutil
from pathlib import Path 
import openai
import uuid
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_redis import RedisChatMessageHistory
import redis

from langchain_core.globals import set_debug, set_verbose
# set_debug(True)

# Testing Session
in_memory_sessions = {}  # Temporary in-memory session storage for testing

# FastAPI 인스턴스 생성
# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global collection
    load_milvus_collection()  # Load the default collection at startup
    yield  # Yield to run the application
    # Shutdown logic
    pass  # You can add any cleanup code here, if needed

app = FastAPI(lifespan=lifespan)

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수 읽기
milvus_url = os.getenv("MILVUS_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
REDIS_URL = os.getenv("REDIS_URL")

# OpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o")
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-large")
PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    Assistant:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# Milvus에 연결
connections.connect(host=milvus_url.split(":")[0], port=milvus_url.split(":")[1])
print("Milvus에 성공적으로 연결되었습니다.")

# Milvus 설정 (Collection과 Field 설정)
DEFAULT_COLLECTION_NAME = "chat_korea_univ"
collection = None  # Global variable to store the loaded collection

# Milvus configuration defaults
DEFAULT_PK_FIELD = "doc_id"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_FILE_ID_FIELD = "file_id"

# Pydantic model for custom collection loading
class LoadCollectionRequestModel(BaseModel):
    collection_name: str

# Function to load the collection
@app.post("/load_milvus_collection/")
def load_milvus_collection(request: LoadCollectionRequestModel = None):
    global collection  # Declare collection as a global variable
    collection_name = request.collection_name if request else DEFAULT_COLLECTION_NAME

    # Check if the collection exists
    if has_collection(collection_name):
        collection = Collection(collection_name)
        print(f"Collection '{collection_name}' loaded successfully.")
        return {"message": f"Collection '{collection_name}' loaded successfully."}
    
    # If the collection doesn't exist, raise an error
    raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' does not exist.")

# Pydantic model for dense index settings
class DenseIndexModel(BaseModel):
    index_type: str = "IVF_FLAT"  # Default index type for Milvus
    metric_type: str = "L2"  # Default metric type for similarity search
    params: dict = {"nlist": 128}  # Default index parameters (nlist controls partitioning)

    class Config:
        schema_extra = {
            "example": {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            },
            "description": {
                "index_type": "Type of index to use for dense vector search. Default is 'IVF_FLAT'. Other options include 'FLAT'.",
                "metric_type": "The metric type to use for similarity. Default is 'L2' (Euclidean distance).",
                "params": "Additional parameters for the index. For example, 'nlist' controls the number of clusters in IVF-based indices."
            }
        }

# Pydantic model for collection creation request
class CreateCollectionRequestModel(BaseModel):
    collection_name: str  # Collection name provided by the user
    dense_index: DenseIndexModel = DenseIndexModel()  # Dense index settings with default values

# Function to create a new collection using the Pydantic model
@app.post("/create_milvus_collection/")
def create_milvus_collection(request: CreateCollectionRequestModel):
    collection_name = request.collection_name
    dense_index = request.dense_index

    if has_collection(collection_name):
        raise HTTPException(status_code=400, detail=f"Collection '{collection_name}' already exists.")
    
    print(f"Collection '{collection_name}' does not exist. Creating new collection.")
    
    # Define fields for the collection schema
    fields = [
        FieldSchema(
            name=DEFAULT_PK_FIELD,
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072),
        FieldSchema(name=DEFAULT_TEXT_FIELD, dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name=DEFAULT_FILE_ID_FIELD, dtype=DataType.VARCHAR, max_length=1000),
    ]
    
    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
    collection = Collection(
        name=collection_name, schema=schema, consistency_level="Strong"
    )
    
    # Use provided dense index parameters
    index_params = {
        "index_type": dense_index.index_type,
        "metric_type": dense_index.metric_type,
        "params": dense_index.params
    }
    collection.create_index("vector", index_params)

    print(f"Collection '{collection_name}' created successfully with index {dense_index.index_type}.")
    return {"message": f"Collection '{collection_name}' created successfully with index {dense_index.index_type}."}
 
def query_all_documents(collection, batch_size=100):
    offset = 0
    all_texts  = []

    while True:
        # Query the collection in batches
        query_results = collection.query(
            expr="",
            output_fields=["text", "file_name"],  # Add more fields if needed
            offset=offset,
            limit=batch_size
        )
        
        # Break the loop if no more results are returned
        if not query_results:
            break
        
        # Extract the "text" field from each document and add it to the list
        all_texts.extend([doc["text"] for doc in query_results])
        
        # Increment the offset for the next batch
        offset += batch_size

    return all_texts 


"""
File Upload, Embedding and Insertion Area
"""

# Global list to store all texts for BM25 embedding
corpus_list = []

# Define the request model with chunk_size and chunk_overlap as parameters
class FileUploadModel(BaseModel):
    chunk_size: int = Field(default=1000, description="Size of each text chunk for splitting.")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks during text splitting.")
    files: List[UploadFile] = Field(..., description="List of text files (.txt) or zip files containing text files to upload and process.")

# Helper function to process a single file (txt) and store the split texts in the global list
async def process_txt_file(file_path: Path, chunk_size: int, chunk_overlap: int):
    # Try reading the file in UTF-8, fallback to a more generic encoding if an error occurs
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        # If utf-8 fails, try reading the file with a different encoding (latin-1)
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    file_id = file_path.name  # Default file name without encoding transformations

    # Split the text using chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)

    # Append tuples of (text, file_id) to the global corpus list
    for t in texts:
        corpus_list.append((t, file_id))  # Now storing both text and file_id

# After processing all files, we generate embeddings for the entire corpus and insert them into Milvus
def insert_texts_into_milvus(collection):
    entities = []
    corpus_list_with_only_texts = [t[0] for t in corpus_list]
    
    for text, file_id in corpus_list:
        entity = {
            "vector": OpenAIEmbeddings(openai_api_key=openai_api_key).embed_documents([text])[0],
            "text": text,
            "file_id": file_id  # Include file_id during insertion
        }
        entities.append(entity)

    # Insert the entities into Milvus
    collection.insert(entities)
    collection.load()  # Load the updated collection

# Endpoint to upload multiple files and process them with chunk parameters
@app.post("/upload/")
async def upload_files_or_zip(request: FileUploadModel):
    try:
        for file in request.files:
            if file.filename.endswith(".txt"):
                temp_file_path = f"/tmp/{file.filename}"
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(await file.read())
                await process_txt_file(Path(temp_file_path), request.chunk_size, request.chunk_overlap)
            elif file.filename.endswith(".zip"):
                temp_dir = Path("/tmp/uploaded_zip")
                temp_dir.mkdir(parents=True, exist_ok=True)

                zip_file_path = temp_dir / file.filename
                with open(zip_file_path, "wb") as temp_zip:
                    temp_zip.write(await file.read())

                extracted_dir = temp_dir / file.filename.replace(".zip", "")
                shutil.unpack_archive(str(zip_file_path), str(extracted_dir), "zip")

                for txt_file in extracted_dir.glob("**/*.txt"):
                    await process_txt_file(txt_file, request.chunk_size, request.chunk_overlap)

        # Now, insert all split texts and their embeddings into Milvus, including file_id
        insert_texts_into_milvus(collection)

        return {"message": "Files uploaded and processed successfully into Milvus."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
"""
Query Area
"""

# Define a query model for incoming queries with additional parameters
class QueryModel(BaseModel):
    query: str
    collection_name: str = Field(default="chat_korea_univ", description="The Milvus collection name to search in.")
    search_type: str = Field(default="similarity", description="The type of search in Milvus (e.g., 'similarity', 'hybrid').")
    search_kwargs: Dict[str, int] = Field(default={"k": 5}, description="The search parameters for Milvus (e.g., {'k': 5}).")
    bm25_k: int = Field(default=5, description="The number of documents to retrieve using BM25 retriever.")
    ensemble_weights: List[float] = Field(default=[0.5, 0.5], description="The weights for EnsembleRetriever (e.g., [0.5, 0.5]).")

# Dependency to create or retrieve a session ID
def get_session_id(session_id: Optional[str] = Header(None)):
    # Check if session_id is provided; if not, create a new one
    if session_id is None:
        session_id = str(uuid.uuid4())
        in_memory_sessions[session_id] = {"created": True}
    elif session_id not in in_memory_sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    return session_id

def format_docs(docs):
    formatted_text = "\n\n".join(doc.page_content for doc in docs)
    return formatted_text

# 사용자 정의 RedisChatMessageHistory 클래스에 만료 시간 추가
class ExpiringRedisChatMessageHistory(RedisChatMessageHistory):
    def __init__(self, session_id, redis_url, ttl=1800):
        super().__init__(session_id=session_id, redis_url=redis_url)
        self.ttl = ttl  # 만료 시간 설정

    def add_user_message(self, message):
        super().add_user_message(message)
        self._set_expiry()

    def add_ai_message(self, message):
        super().add_ai_message(message)
        self._set_expiry()

    def _set_expiry(self):
        # Redis에 저장된 세션 데이터에 만료 시간 설정
        redis_client = redis.from_url(self.redis_url)
        redis_client.expire(f"history:{self.session_id}", self.ttl)

# 테스트 함수
@app.get("/test_redis")
def test_redis():
    # 만료 시간 30분으로 설정한 RedisChatMessageHistory 객체 초기화
    history = ExpiringRedisChatMessageHistory(session_id="user_123", redis_url=REDIS_URL, ttl=1800)

    # 메시지 추가
    history.add_user_message("Hello, AI assistant!")
    history.add_ai_message("Hello! How can I assist you today?")

    # 메시지 조회
    print("Chat History:")
    for message in history.messages:
        print(f"{type(message).__name__}: {message.content}")
        
# Query Endpoint using Milvus Hybrid Search Retriever and LangChain RAG chain
@app.post("/query/")
async def query_langchain(query: QueryModel, session_id: str = Depends(get_session_id)):
    try:
        # Getting History
        history = ExpiringRedisChatMessageHistory(session_id=session_id, redis_url=REDIS_URL, ttl=1800)
        # Add user message to Redis history
        history.add_user_message(query.query)
        
        # Setup retriever
        milvus_vector_store = Milvus(
            embedding,
            connection_args={"host": milvus_url.split(":")[0], "port": milvus_url.split(":")[1]},
            collection_name=query.collection_name,  # Use collection name from query
        )
        
        milvus_retriever = milvus_vector_store.as_retriever(search_type=query.search_type, search_kwargs=query.search_kwargs)
    
        # Initialize the BM25 retriever
        texts = query_all_documents(collection)
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = query.bm25_k  # Set k from query parameters
        
        # Set up the EnsembleRetriever with custom weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, milvus_retriever], 
            weights=query.ensemble_weights  # Use weights from query parameters
        )
        
        # Beautifier function to clean up and format retrieved documents
        def beautify_docs(docs):
            beautified = []
            for doc in docs:
                # Extract metadata and content
                metadata = "\n".join([f"{k}: {v}" for k, v in doc.metadata.items()]) if doc.metadata else "No metadata"
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content  # Truncate long content for readability
                
                # Format for readability
                beautified.append(f"Metadata:\n{metadata}\n\nContent:\n{content}\n{'-'*50}")
            return "\n".join(beautified)

        # Define a logging step to print the retrieved documents
        log_retrieved_docs = RunnableLambda(lambda x: (print("Retrieved Docs:", beautify_docs(x)), x)[1])

        # Define the RAG chain
        rag_chain = (
            {"context": ensemble_retriever
            | log_retrieved_docs
            | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        rag_chain.get_graph().print_ascii()
        answer = rag_chain.invoke(query.query)
        rag_chain.max_tokens_limit = 
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying with Langchain: {str(e)}")


"""
DELETE Area
"""

# Define a model for deleting records by file_id from a specific collection
class DeleteRecordModel(BaseModel):
    collection_name: str
    file_id: str

# Endpoint to delete records from a specific collection based on file_id
@app.delete("/delete_record/")
async def delete_record(delete_model: DeleteRecordModel):
    try:
        # Check if the collection exists
        if not has_collection(delete_model.collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{delete_model.collection_name}' does not exist.")

        # Load the collection
        collection = Collection(delete_model.collection_name)

        # Delete records where the file_id matches
        expr = f"file_id == '{delete_model.file_id}'"
        collection.delete(expr)

        return {"message": f"Records with file_id '{delete_model.file_id}' have been successfully deleted from collection '{delete_model.collection_name}'."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting records: {str(e)}")

# Endpoint to delete an entire collection
@app.delete("/delete_collection/")
async def delete_collection(collection_name: str):
    try:
        # Check if the collection exists
        if has_collection(collection_name):
            # Drop the collection if it exists
            drop_collection(collection_name)
            return {"message": f"Collection '{collection_name}' has been successfully deleted."}
        else:
            return {"message": f"Collection '{collection_name}' does not exist."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

# File deletion endpoint (triggered by Lambda when a file is deleted from S3)
@app.post("/delete_file/")
async def delete_file(file_name: str, collection_name: str = "document_collection"):
    try:
        # Check if collection exists
        if not has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' does not exist.")

        # Delete file data from Milvus by file_id
        collection = Collection(collection_name)
        expr = f"file_id == '{file_name}'"
        collection.delete(expr)

        return {"message": f"File '{file_name}' deleted from collection '{collection_name}'."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    
# FOR THE FUTURE USE
# Setup the Milvus Hybrid Search Retriever
def setup_milvus_hybrid_retriever():
    dense_embedding_func = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = query_all_documents(collection)
    sparse_embedding_func = BM25SparseEmbedding(texts)
    
    # Milvus hybrid search retriever
    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "IP", "params": {}}
    
    dense_field = "dense_vector"
    sparse_field = "sparse_vector"
    text_field = "text"
    
    retriever = MilvusCollectionHybridSearchRetriever(
        collection=collection,
        rerank=WeightedRanker(0.5, 0.5),
        anns_fields=[dense_field, sparse_field],
        field_embeddings=[dense_embedding_func, sparse_embedding_func],
        field_search_params=[dense_search_params, sparse_search_params],
        top_k=3,
        text_field=text_field,
    )
    
    return retriever

