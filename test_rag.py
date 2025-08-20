import os
import importlib
import rag

importlib.reload(rag)

print("Fisier rag:", rag.__file__)
print("Are build_index?", hasattr(rag, "build_index"))
print("Are retrieve?", hasattr(rag, "retrieve"))

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Setează variabila de mediu OPENAI_API_KEY in acest terminal.")

print("\nConstruiesc indexul...")
count = rag.build_index()
print("Cărți indexate:", count)

print("\nCăutare test:")
results = rag.retrieve("vreau o carte despre magie și prietenie", 3)
for r in results:
    print("-", r["metadata"]["title"], "=>", r["distance"], "\n", r["document"], "\n")
