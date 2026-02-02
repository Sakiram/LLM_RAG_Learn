from langchain_openai import OpenAIEmbeddings
from app.config import OPENAI_API_KEY, EMBED_MODEL

def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=OPENAI_API_KEY
    )