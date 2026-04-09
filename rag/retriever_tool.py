import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI API 키를 환경 변수로 설정
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        embeddings = OpenAIEmbeddings()
        vector_store = ElasticsearchStore(
            es_url=ELASTICSEARCH_URL,
            index_name="llm_*_index", ################# 수정
            embedding=embeddings,
        )
        _retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return _retriever

@tool
def retriever_tool(query: str) -> List:
    """
    Retrieves relevant documents from the Elasticsearch vector store based on a query.
    
    This tool should be used to find information within the pre-indexed knowledge base.
    Input should be a single string query.
    """
    retriever = _get_retriever()
    results = retriever.invoke(query)

    formatted_results = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
    return formatted_results