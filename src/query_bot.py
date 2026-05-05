from typing import List, Tuple
from functools import lru_cache

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "ipvc_courses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3:latest"


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
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


def detect_school_code(question: str):
    question_lower = question.lower()

    school_codes = ["esa", "ese", "estg", "esce", "ess", "esdl"]
    for code in school_codes:
        if code in question_lower:
            return code.upper()

    return None


def is_courses_by_school_question(question: str) -> bool:
    question_lower = question.lower()
    school_code = detect_school_code(question)

    if school_code is None:
        return False

    keywords = [
        "que cursos",
        "quais são os cursos",
        "cursos existem",
        "cursos da",
        "cursos do"
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


def get_courses_by_school(school_code: str) -> Tuple[List[str], List[dict]]:
    collection = get_collection()

    results = collection.get(
        where={
            "$and": [
                {"type": "structured_course"},
                {"escola": school_code}
            ]
        }
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    return documents, metadatas


def search_relevant_context(question: str, n_results: int = 6) -> Tuple[List[str], List[dict]]:
    if is_school_question(question):
        return get_all_school_documents()

    if is_courses_by_school_question(question):
        school_code = detect_school_code(question)
        return get_courses_by_school(school_code)

    collection = get_collection()
    embedding_model = get_embedding_model()

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
- Não omitas elementos do contexto quando a pergunta pedir uma listagem completa.

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