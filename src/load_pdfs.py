from pathlib import Path
from typing import Union
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)


def load_all_pdfs(folder_path: Union[str, Path]) -> list[dict]:
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")

    pdf_files = list(folder_path.glob("*.pdf"))
    documents = []

    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_file)
            documents.append({
                "source": pdf_file.name,
                "text": text
            })
        except Exception as e:
            print(f"Erro ao ler {pdf_file.name}: {e}")

    return documents


if __name__ == "__main__":
    path = Path("data/nao_estruturados")
    docs = load_all_pdfs(path)

    print(f"Foram carregados {len(docs)} PDFs.")
    for doc in docs[:2]:
        print(f"\nFonte: {doc['source']}")
        print(doc["text"][:500])