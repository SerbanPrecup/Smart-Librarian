import json

def load_books(path: str = "data/book_summaries.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_summary_by_title(title: str, path: str = "data/book_summaries.json") -> str:
    books = load_books(path)
    for b in books:
        if b["title"].strip().lower() == title.strip().lower():
            return b.get("full_summary") or "No detailed summary available."
    return "No detailed summary found for this title."
