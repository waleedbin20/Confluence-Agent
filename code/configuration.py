from json import tool
import os
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import AnnSearchRequest, Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, RRFRanker, connections
from langchain_community.document_loaders import ConfluenceLoader
from pymilvus import model
from pymilvus import utility
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import MessagesPlaceholder
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import model
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
from dotenv import load_dotenv

load_dotenv()

connections.connect("default", uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))

if not os.getenv("MILVUS_URI") or not os.getenv("MILVUS_TOKEN"):
    raise ValueError("Milvus connection parameters are not set properly.")

print("##### Starting model ###")

# EMBEDDINGS = BGEM3EmbeddingFunction(
#     device='cpu', 
#     use_fp16=False 
# )
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in your environment variables.")

EMBEDDINGS = model.dense.OpenAIEmbeddingFunction(
    model_name='text-embedding-3-large', 
    dimensions=620,
    api_key=openai_api_key
)


COLLECTION_NAME = "CONFLUENCE_KNOWLEDGE_BASE"


class ConfluenceConnection:
    url: str
    token: str
    username: str
    space_key: str
    include_attachments: bool
    limit: int
    ocr_languages: str
    max_pages: int


def create_collection():

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR,
                    is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10_000),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDINGS.dim),  # Fixed to use EMBEDDINGS.dim directly
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=5012)
    ]

    schema = CollectionSchema(fields, "")
    col = Collection(COLLECTION_NAME, schema)

    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "FLAT", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)
    col.load()


def load_confluence(confluence) -> None:
    """
    Get data from confluence and load to a vector db.
    Args:
        confluence: Confluence settings

    Returns:

    """
    if utility.has_collection(COLLECTION_NAME):
        print(f"✅ Collection '{COLLECTION_NAME}' already exists. Skipping load.")
        return
    loader = ConfluenceLoader(
        url=confluence.url,
        api_key=confluence.token,  # Fixed to use token for authentication
        username=confluence.username,
        space_key=confluence.space_key,
        include_attachments=confluence.include_attachments,
        limit=confluence.limit,
        ocr_languages=confluence.ocr_languages,
        max_pages=confluence.max_pages
    )

    docs = loader.load()
        
    chunks = RecursiveCharacterTextSplitter().split_documents(docs)

    print(f"Split into {len(chunks)} chunks.")

    create_collection()
    insert_doc_to_collection(chunks, collection_name=COLLECTION_NAME)



def insert_doc_to_collection(
        documents: List[Document],
        collection_name: str
):
    if not documents:
        print("No documents to insert.")
        return
        
    col = Collection(collection_name)
    contents = []
    titles = []
    sources = []
    for doc in documents:
        contents.append(doc.page_content)
        titles.append(doc.metadata["title"])
        sources.append(doc.metadata["source"])

    # Get embeddings - OpenAI only returns dense vectors
    try:
        dense_vectors = EMBEDDINGS.encode_documents(contents)
        print("Embeddings:", dense_vectors)
        print("Dim:", EMBEDDINGS.dim, dense_vectors[0].shape)


        # OpenAI embeddings are returned directly as a list, not in a dict
        if isinstance(dense_vectors, dict):
            dense_vectors = dense_vectors.get("dense", [])
        
        # Create empty sparse vectors since we don't have them
        sparse_vectors = [[]] * len(contents)
        
        print(f"Generated {len(dense_vectors)} dense vectors")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # Check if any of the lists are empty
    if not contents or not dense_vectors or not titles or not sources:
        print("Warning: One or more data fields are empty.")
        return
    
    # Ensure all lists are the same length
    min_length = min(len(contents), len(dense_vectors), len(titles), len(sources))
    if min_length == 0:
        print("No valid data to insert.")
        return
        
    contents = contents[:min_length]
    sparse_vectors = sparse_vectors[:min_length]
    dense_vectors = dense_vectors[:min_length]
    titles = titles[:min_length]
    sources = sources[:min_length]
    
    entities = [contents, sparse_vectors, dense_vectors, titles, sources]

    col.insert(entities)
    col.flush()
    
    
    
@tool
def hybrid_search(
        query: str
) -> List[Dict[str, Any]]:
    """
    Perform a hybrid search (semantic and keyword) on documents in a Milvus collection.

    Args:
        query: User query text to search for.
        limit: retrieve this many documents
    Returns:
        A list of documents matching the query, with their content and metadata.
    """
    limit = 15
    if not utility.has_collection(COLLECTION_NAME):
        return [{"error": f"Collection '{COLLECTION_NAME}' does not exist."}]

    col = Collection(COLLECTION_NAME)

    try:
        # Get embeddings from OpenAI
        query_embeddings = EMBEDDINGS.encode_queries([query])
        
        # For OpenAI embeddings (dense only)
        if not isinstance(query_embeddings, dict):
            # Create empty sparse vector since we only have dense
            sparse_vector = []
            dense_vector = query_embeddings  # OpenAI returns dense vectors directly
            
            # Log what's happening
            print(f"Using dense search only with vector dimension: {len(dense_vector[0])}")
        else:
            # For hybrid models with both sparse and dense
            sparse_vector = query_embeddings.get("sparse", [])
            dense_vector = query_embeddings.get("dense", [])

        # Perform search
        search_params = {"metric_type": "IP"}
        
        # If we have sparse vectors, do hybrid search
        if sparse_vector and len(sparse_vector[0]) > 0:
            sparse_req = AnnSearchRequest(sparse_vector, "sparse_vector", search_params, limit=limit)
            dense_req = AnnSearchRequest(dense_vector, "dense_vector", search_params, limit=limit)
            res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                   limit=limit, output_fields=['text', "source", "title"])
        else:
            # Do dense-only search if sparse vectors aren't available
            res = col.search(dense_vector, "dense_vector", search_params, 
                            limit=limit, output_fields=['text', "source", "title"])
        
        # Process results - ensure we're handling results properly regardless of search type
        if isinstance(res, list) and len(res) > 0:
            # For hybrid search result
            results = res[0]
        else:
            # For regular search result
            results = res
            
        documents = []
        for hit in results:
            documents.append({
                "content": hit.fields["text"],
                "source": hit.fields["source"],
                "title": hit.fields["title"],
                "score": hit.distance
            })
        
        return documents
    
    except Exception as e:
        print(f"Search error: {e}")
        return [{"error": f"Error performing search: {str(e)}"}]


# Create the agent with tools
def create_agent_executor(openai_api_key: str, model_name: str = "gpt-4o"):
    """
    Create an agent executor with hybrid search

    Args:
        openai_api_key: OpenAI API key.
        model_name: Name of the OpenAI model to use.

    Returns:
        An AgentExecutor that can use the tools.
    """
     # Load system prompt from file
    prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    with open(prompt_file_path, "r") as file:
        system_prompt = file.read()
        
    llm = ChatOpenAI(
        temperature=0,
        model=model_name,
        api_key=openai_api_key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="output"
    )
    tools = [hybrid_search]

    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

    return agent_executor