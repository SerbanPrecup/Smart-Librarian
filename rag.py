import os, json
from typing import List, Dict, Any
from chromadb import PersistentClient
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


MODEL_EMBED = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def load_books(path: str = "data/book_summaries.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def get_chroma_collection(path: str = "chroma_db" , name: str = "books"):
    client  = PersistentClient(path=path)
    try:
        col = client.get_collection(name=name)
    except Exception:
        col = client.create_collection(name=name)
    return col

def _doc_for_index(book: Dict[str, Any]) -> str:
    themes = ", ".join(book.get("themes", []))
    return f"Titlu: {book['title']}\nRezumat scurt: {book['short_summary']}\nTeme: {themes}"

def build_index(data_path="data/book_summaries.json", db_path="chroma_db", collection_name="books"):
    books = load_books(data_path)
    collection = get_chroma_collection(db_path, collection_name)
    client = OpenAI()

    try:
        collection.delete(where={})
    except Exception:
        pass

    ids, docs, metas, embs = [], [], [], []
    for i, b in enumerate(books, start=1):
        text = _doc_for_index(b)
        emb = client.embeddings.create(model=MODEL_EMBED, input=text).data[0].embedding
        ids.append(f"book-{i}")
        docs.append(text)
        metas.append({"title": b["title"]})
        embs.append(emb)

    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(ids)

def build_index(data_path="data/book_summaries.json", db_path="chroma_db", collection_name="books"):
    books = load_books(data_path)
    collection = get_chroma_collection(db_path, collection_name)
    client = OpenAI()

    try:
        collection.delete(where={})
    except Exception:
        pass

    ids, docs, metas, embs = [], [], [], []
    for i, b in enumerate(books, start=1):
        text = _doc_for_index(b)
        emb = client.embeddings.create(model=MODEL_EMBED, input=text).data[0].embedding
        ids.append(f"book-{i}")
        docs.append(text)
        metas.append({"title": b["title"]})
        embs.append(emb)

    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(ids)

def retrieve(query: str, n_results: int = 4, db_path="chroma_db", collection_name="books"):
    collection = get_chroma_collection(db_path, collection_name)
    client = OpenAI()
    q_emb = client.embeddings.create(model=MODEL_EMBED, input=query).data[0].embedding
    res = collection.query(query_embeddings=[q_emb], n_results=n_results)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i] if "distances" in res else None
        })
    return hits