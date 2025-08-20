
import os, json
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
from rag import retrieve
from tools import get_summary_by_title

load_dotenv()
console = Console()
MODEL_CHAT = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Return the full summary for a given book title.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"]
            }
        }
    }
]

def detect_language(text: str) -> str:
    """
    Întoarce 'ro' dacă textul pare română, altfel 'en'.
    """
    if any(ch in text for ch in "ăîâșțĂÎÂȘȚ"):
        return "ro"

    ro_keywords = ["si", "este", "carte", "prietenie", "magie", "vreau", "recomandare"]
    if any(word in text.lower() for word in ro_keywords):
        return "ro"
    return "en"

def chat_once(question: str) -> str:
    client = OpenAI()
    hits = retrieve(question, n_results=4)

    context = "\n\n---\n\n".join(
        f"[{h['metadata'].get('title','N/A')}]\n{h['document']}" for h in hits
    )

    lang = detect_language(question)

    if lang == "ro":
        system = (
            "Ești Smart Librarian. Analizează candidații și alege EXACT o carte. "
            "Apoi apelează tool-ul get_summary_by_title cu titlul exact. "
            "Răspunsul final trebuie să fie în LIMBA ROMÂNĂ și să includă: "
            "1) Recomandarea (titlu + de ce se potrivește) 2) Rezumatul detaliat (din tool)."
        )
    else:
        system = (
            "You are Smart Librarian. Analyze the candidates and pick EXACTLY one book. "
            "Then CALL the tool get_summary_by_title with the exact title. "
            "Final answer must be in ENGLISH and include: "
            "1) Recommendation (title + why it fits) 2) Detailed summary (from the tool)."
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nCandidates:\n{context}"}
    ]

    first = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.3
    )
    msg = first.choices[0].message
    tool_calls = getattr(msg, "tool_calls", []) or []

    messages.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ]
    })

    if tool_calls:
        for tc in tool_calls:
            if tc.function.name == "get_summary_by_title":
                args = json.loads(tc.function.arguments or "{}")
                title = args.get("title", "")
                summary = get_summary_by_title(title)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "get_summary_by_title",
                    "content": summary
                })

        final = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=messages,
            temperature=0.2
        )
        return final.choices[0].message.content

    return msg.content or "No useful response."

def main():
    console.print(Panel.fit("🎓 Smart Librarian – RAG + Tool Calling", subtitle="scrie 'exit' pentru a ieși"))
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Setează OPENAI_API_KEY (în .env sau env).[/red]")
        return
    while True:
        console.print("\n[bold]Tu:[/bold] ", end="")
        q = input().strip()
        if q.lower() in {"exit", "quit", "iesire", "ieșire"}:
            break
        if not q:
            continue
        try:
            ans = chat_once(q)
            console.print(Panel.fit(ans, title="Răspuns"))
        except Exception as e:
            console.print(f"[red]Eroare: {e}[/red]")

if __name__ == "__main__":
    main()
