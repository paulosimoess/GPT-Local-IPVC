from typing import List, Tuple

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "ipvc_courses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3:latest"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)


def is_school_question(question: str) -> bool:
    question_lower = question.lower()
    keywords = [
        "que escolas",
        "quais são as escolas",
        "escolas do ipvc",
        "escolas existem",
        "quais as escolas"
    ]
    return any(keyword in question_lower for keyword in keywords)


def get_all_school_documents() -> Tuple[List[str], List[dict]]:
    collection = get_collection()

    results = collection.get(
        where={"type": "structured_school"}
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    return documents, metadatas


def search_relevant_context(question: str, n_results: int = 6) -> Tuple[List[str], List[dict]]:
    if is_school_question(question):
        return get_all_school_documents()

    collection = get_collection()
    query_embedding = embedding_model.encode([question]).tolist()[0]

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
- Nunca uses conhecimento externo.
- Se a resposta não estiver claramente no contexto, responde apenas: "Não encontrei essa informação nos documentos disponíveis."
- Não faças suposições.
- Não acrescentes exemplos inventados.
- Responde em português de Portugal.
- Sê claro e objetivo.
- Se a pergunta pedir listagens, usa tópicos.
- Se o contexto contiver uma lista completa, apresenta a lista completa.

Contexto:
{context_text}

Pergunta:
{question}

Resposta:
""".strip()


def ask_bot(question: str) -> Tuple[str, List[dict], List[str]]:
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

    return response["message"]["content"], metadatas, context_chunks


if __name__ == "__main__":
    print("Bot IPVC pronto. Escreve 'sair' para terminar.\n")

    while True:
        question = input("Pergunta: ").strip()

        if question.lower() in {"sair", "exit", "quit"}:
            print("Até logo.")
            break

        try:
            answer, metadatas, context_chunks = ask_bot(question)

            print("\nFontes recuperadas:")
            for i, meta in enumerate(metadatas, start=1):
                source = meta.get("source", "desconhecida")
                doc_type = meta.get("type", "desconhecido")
                print("{0}. {1} ({2})".format(i, source, doc_type))

            print("\nResposta:")
            print(answer)
            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print("\nErro: {0}\n".format(e))