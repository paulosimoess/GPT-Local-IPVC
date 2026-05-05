from pathlib import Path
from typing import List, Dict, Union

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from load_structured_data import load_courses_csv
from load_pdfs import load_all_pdfs


CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "ipvc_courses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def load_generic_csv(csv_path: Union[str, Path]) -> List[Dict]:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError("CSV não encontrado: {0}".format(csv_path))

    df = pd.read_csv(csv_path)
    return df.fillna("").to_dict(orient="records")


def course_to_text(course: Dict) -> str:
    return (
        "Curso: {0}\n"
        "Escola: {1}\n"
        "Grau: {2}\n"
        "Duração: {3} anos\n"
        "ECTS: {4}\n"
        "Regime: {5}\n"
        "Local: {6}\n"
        "Área: {7}\n"
        "Saídas profissionais: {8}\n"
        "Descrição: {9}\n"
        "Resumo: {10}\n"
        "Interesses relacionados: {11}\n"
        "Palavras-chave: {12}\n"
        "Fonte: {13}"
    ).format(
        course.get("curso", ""),
        course.get("escola", ""),
        course.get("grau", ""),
        course.get("duracao_anos", ""),
        course.get("ects", ""),
        course.get("regime", ""),
        course.get("local", ""),
        course.get("area", ""),
        course.get("saidas_profissionais", ""),
        course.get("descricao", ""),
        course.get("resumo", ""),
        course.get("interesses_relacionados", ""),
        course.get("palavras_chave", ""),
        course.get("fonte", "")
    )


def school_to_text(school: Dict) -> str:
    return (
        "Escola: {0}\n"
        "Sigla: {1}\n"
        "Local: {2}\n"
        "Descrição: {3}"
    ).format(
        school.get("escola", ""),
        school.get("sigla", ""),
        school.get("local", ""),
        school.get("descricao", "")
    )


def build_documents() -> List[Dict]:
    documents = []

    # Cursos
    courses_path = Path("data/estruturados/cursos_ipvc.csv")
    courses = load_courses_csv(courses_path)

    for idx, course in enumerate(courses):
        text = course_to_text(course)
        documents.append({
            "id": "course_{0}".format(idx),
            "text": text,
            "metadata": {
                "type": "structured_course",
                "source": course.get("fonte", ""),
                "curso": course.get("curso", ""),
                "escola": course.get("escola", ""),
                "grau": course.get("grau", ""),
                "area": course.get("area", "")
            }
        })

    # Escolas
    schools_path = Path("data/estruturados/escolas_ipvc.csv")
    schools = load_generic_csv(schools_path)

    for idx, school in enumerate(schools):
        text = school_to_text(school)
        documents.append({
            "id": "school_{0}".format(idx),
            "text": text,
            "metadata": {
                "type": "structured_school",
                "source": "escolas_ipvc.csv",
                "escola": school.get("escola", ""),
                "sigla": school.get("sigla", ""),
                "local": school.get("local", "")
            }
        })

    # PDFs
    pdfs_path = Path("data/nao_estruturados")
    pdf_docs = load_all_pdfs(pdfs_path)

    doc_counter = 0
    for pdf in pdf_docs:
        chunks = chunk_text(pdf["text"])

        for chunk in chunks:
            documents.append({
                "id": "pdf_{0}".format(doc_counter),
                "text": chunk,
                "metadata": {
                    "type": "pdf_chunk",
                    "source": pdf["source"]
                }
            })
            doc_counter += 1

    return documents


def create_vector_database() -> None:
    print("A carregar modelo de embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("A preparar documentos...")
    documents = build_documents()
    print("Total de documentos/blocos: {0}".format(len(documents)))

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    existing_collections = client.list_collections()
    existing_names = [collection.name for collection in existing_collections]

    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    print("A gerar embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("A guardar no ChromaDB...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("Base vetorial criada com sucesso.")
    print("Coleção: {0}".format(COLLECTION_NAME))
    print("Documentos guardados: {0}".format(len(ids)))


if __name__ == "__main__":
    create_vector_database()