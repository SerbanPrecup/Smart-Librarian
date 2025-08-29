# Smart Librarian (CLI)

Chatbot that recommends **a single book** based on the user's questions, using:
- **RAG** (ChromaDB + OpenAI embeddings)
- **OpenAI Chat API**
- Local tool `get_summary_by_title()` for the full summary.

> According to the assignment requirements: “Smart Librarian – AI with RAG + Tool Completion”.

⚡ **Note**: The chatbot works in **both English and Romanian**, depending on the language of the prompt.

## Requirements
- Python **3.10+**
- `OPENAI_API_KEY` in `.env`
- Dependencies from `requirements.txt` (openai, chromadb, python-dotenv, rich)

## Installation
    pip install -r requirements.txt

## `.env` Configuration
    OPENAI_API_KEY=sk-...

*(Optional)*  
    OPENAI_CHAT_MODEL=gpt-4o-mini  
    OPENAI_EMBED_MODEL=text-embedding-3-small

## Index Initialization (ChromaDB)
Build the collection with embeddings from `data/book_summaries.json`:
    
    python test_rag.py

## Run (CLI)
    python app_cli.py

Exit with: `exit` | `quit` | `iesire` | `ieșire`.

## Example Questions
- “I want a book about freedom and social control.”  
- “What do you recommend if I love fantastic stories?”  
- “What is 1984?”  
- „Vreau o carte despre libertate și control social.”  
- „Ce-mi recomanzi dacă iubesc poveștile fantastice?”  
- „Ce este 1984?”

## Minimal Structure
    app_cli.py        # CLI + GPT integration + tool calling
    rag.py            # embeddings + ChromaDB (build & retrieve)
    tools.py          # function get_summary_by_title(title)
    test_rag.py       # script to initialize/verify RAG
    requirements.txt
    data/book_summaries.json  # 10+ books with summaries (short/full, themes)
    chroma_db/        # (generated) persistent vector database

## Note
- Vector store: **ChromaDB** (not OpenAI vector store), with **OpenAI embeddings**, semantic retrieval, and tool calling for the full summary, according to the requirements.
