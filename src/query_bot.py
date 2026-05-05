from typing import List, Tuple

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "ipvc_courses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3:latest"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)


def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def search_relevant_context(question: str, n_results: int = 6) -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    model = get_embedding_model()

    query_embedding = model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return documents, metadatas


def build_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)

    return f"""
És um assistente académico especializado nos cursos do IPVC.

Regras obrigatórias:
- Responde apenas com base no contexto fornecido.
- Não inventes informação.
- Se a informação não estiver no contexto, diz claramente que não foi encontrada.
- Responde em português de Portugal.
- Sê objetivo, claro e útil.
- Quando a pergunta pedir listagens, apresenta em tópicos.
- Quando fizeres recomendações, justifica com base no contexto.

Contexto:
{context_text}

Pergunta:
{question}

Resposta:
""".strip()


def ask_bot(question: str) -> Tuple[str, List[dict]]:
    context_chunks, metadatas = search_relevant_context(question)
    prompt = build_prompt(question, context_chunks)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"], metadatas


if __name__ == "__main__":
    print("Bot IPVC pronto. Escreve 'sair' para terminar.\n")

    while True:
        question = input("Pergunta: ").strip()

        if question.lower() in {"sair", "exit", "quit"}:
            print("Até logo.")
            break

        try:
            answer, metadatas = ask_bot(question)

            print("\nFontes recuperadas:")
            for i, meta in enumerate(metadatas, start=1):
                source = meta.get("source", "desconhecida")
                doc_type = meta.get("type", "desconhecido")
                print(f"{i}. {source} ({doc_type})")

            print("\nResposta:")
            print(answer)
            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f"\nErro: {e}\n")