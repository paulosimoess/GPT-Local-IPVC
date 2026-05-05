from pathlib import Path
from typing import Union
import pandas as pd


def load_courses_csv(csv_path: Union[str, Path]) -> list[dict]:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    records = df.fillna("").to_dict(orient="records")
    return records


if __name__ == "__main__":
    path = Path("data/estruturados/cursos_ipvc.csv")
    courses = load_courses_csv(path)

    print(f"Foram carregados {len(courses)} cursos.")
    for course in courses[:3]:
        print(course)