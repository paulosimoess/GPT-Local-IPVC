from pathlib import Path
from typing import List, Dict

import chromadb
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


def course_to_text(course: Dict) -> str:
    return (
        f"Curso: {course.get('curso', '')}\n"
        f"Escola: {course.get('escola', '')}\n"
        f"Grau: {course.get('grau', '')}\n"
        f"Duração: {course.get('duracao_anos', '')} anos\n"
        f"ECTS: {course.get('ects', '')}\n"
        f"Regime: {course.get('regime', '')}\n"
        f"Local: {course.get('local', '')}\n"
        f"Área: {course.get('area', '')}\n"
        f"Saídas profissionais: {course.get('saidas_profissionais', '')}\n"
        f"Fonte: {course.get('fonte', '')}"
    )


def build_documents() -> List[Dict]:
    documents = []

    # Dados estruturados (CSV)
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

    # Dados não estruturados (PDFs)
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

    # Se já existir, apaga e recria
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