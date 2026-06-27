import subprocess
import sys
from pathlib import Path


def chroma_db_exists() -> bool:
    chroma_path = Path("chroma_db")

    if not chroma_path.exists():
        return False

    files = [
        file
        for file in chroma_path.rglob("*")
        if file.is_file() and file.name != ".gitkeep"
    ]

    return len(files) > 0


def main():
    if not chroma_db_exists():
        print("Base vetorial não encontrada. A criar ChromaDB...")
        subprocess.check_call([sys.executable, "src/build_vector_db.py"])
    else:
        print("Base vetorial já existe. A iniciar aplicação...")

    subprocess.check_call(
        [
            "streamlit",
            "run",
            "src/app.py",
            "--server.address=0.0.0.0",
            "--server.port=8501",
        ]
    )


if __name__ == "__main__":
    main()