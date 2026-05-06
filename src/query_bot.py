from typing import List, Tuple, Optional
from functools import lru_cache
import unicodedata
import re

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


def normalize_question(question: str) -> str:
    return question.strip().lower()


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text


def detect_school_code(question: str) -> Optional[str]:
    question_norm = normalize_text(question)

    school_codes = ["esa", "ese", "estg", "esce", "ess", "esdl"]

    for code in school_codes:
        pattern = r"\b" + re.escape(code) + r"\b"
        if re.search(pattern, question_norm):
            return code.upper()

    return None


def detect_location(question: str) -> Optional[str]:
    question_norm = normalize_text(question)

    known_locations = {
        "ponte de lima": "Ponte de Lima",
        "ponte lima": "Ponte de Lima",
        "viana do castelo": "Viana do Castelo",
        "viana": "Viana do Castelo",
        "valenca": "Valença",
        "melgaco": "Melgaço"
    }

    for location_norm, location_name in known_locations.items():
        if location_norm in question_norm:
            return location_name

    return None


def detect_degree(question: str) -> Optional[str]:
    question_norm = normalize_text(question)

    degree_keywords = {
        "licenciatura": "Licenciatura",
        "licenciaturas": "Licenciatura",
        "mestrado": "Mestrado",
        "mestrados": "Mestrado",
        "pos graduacao": "Pós-graduação",
        "pos graduacoes": "Pós-graduação",
        "pos-graduacao": "Pós-graduação",
        "pos-graduacoes": "Pós-graduação",
        "pós graduação": "Pós-graduação",
        "pós-graduação": "Pós-graduação",
        "pós graduações": "Pós-graduação",
        "pós-graduações": "Pós-graduação",
        "ctesp": "CTeSP",
        "ctesps": "CTeSP",
        "tesp": "CTeSP"
    }

    for keyword, degree in degree_keywords.items():
        if normalize_text(keyword) in question_norm:
            return degree

    return None


def is_recommendation_question(question: str) -> bool:
    question_norm = normalize_text(question)

    keywords = [
        "aconselhas",
        "aconselha",
        "recomendas",
        "recomenda",
        "recomendarias",
        "que curso devo escolher",
        "que curso escolher",
        "qual curso devo escolher",
        "gosto de",
        "interesse em",
        "interessado em",
        "interessada em",
        "quero trabalhar com",
        "quero trabalhar na area",
        "quero trabalhar na área",
        "quero seguir",
        "tenho interesse",
        "o que devo escolher"
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def expand_recommendation_query(question: str) -> str:
    question_norm = normalize_text(question)

    expansions = []

    interest_map = {
        "animais": "animais veterinaria veterinario saude animal cuidados animais producao animal enfermagem veterinaria",
        "animal": "animais veterinaria veterinario saude animal cuidados animais producao animal enfermagem veterinaria",
        "veterinaria": "animais veterinaria veterinario saude animal enfermagem veterinaria",
        "programacao": "programacao informatica software desenvolvimento aplicacoes engenharia informatica tecnologia computadores",
        "programar": "programacao informatica software desenvolvimento aplicacoes engenharia informatica tecnologia computadores",
        "computadores": "informatica computadores redes sistemas programacao tecnologia software",
        "informatica": "informatica programacao software redes sistemas engenharia informatica",
        "saude": "saude enfermagem cuidados saude comunitaria saude mental fisioterapia gerontologia",
        "enfermagem": "saude enfermagem cuidados saude hospitalar comunitaria",
        "gestao": "gestao empresas administracao contabilidade marketing negocios organizacoes",
        "empresas": "gestao empresas administracao contabilidade marketing negocios organizacoes",
        "marketing": "marketing comunicacao vendas gestao comercial",
        "desporto": "desporto atividade fisica treino exercicio saude bem-estar",
        "educacao": "educacao ensino criancas formacao intervencao educativa",
        "criancas": "educacao criancas infancia ensino intervencao educativa",
        "ambiente": "ambiente sustentabilidade agricultura recursos naturais agronomia",
        "agricultura": "agricultura agronomia ambiente sustentabilidade recursos naturais",
        "turismo": "turismo hotelaria gestao turistica patrimonio lazer",
        "design": "design multimedia criatividade comunicacao visual produto digital",
        "jogos": "jogos digitais programacao multimedia design tecnologia",
        "redes": "redes sistemas computadores ciberseguranca informatica infraestrutura",
        "seguranca": "ciberseguranca seguranca informatica redes sistemas"
    }

    for keyword, expansion in interest_map.items():
        if keyword in question_norm:
            expansions.append(expansion)

    if expansions:
        return question + " " + " ".join(expansions)

    return question


def is_school_question(question: str) -> bool:
    question_norm = normalize_text(question)

    keywords = [
        "que escolas",
        "quais sao as escolas",
        "escolas do ipvc",
        "escolas existem",
        "quais as escolas",
        "lista de escolas",
        "mostrar escolas"
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def is_structured_courses_question(question: str) -> bool:
    question_norm = normalize_text(question)

    has_filter = (
        detect_school_code(question) is not None
        or detect_location(question) is not None
        or detect_degree(question) is not None
    )

    if not has_filter:
        return False

    keywords = [
        "curso",
        "cursos",
        "formacao",
        "formacoes",
        "licenciatura",
        "licenciaturas",
        "mestrado",
        "mestrados",
        "pos graduacao",
        "pos graduacoes",
        "ctesp",
        "ctesps",
        "quais",
        "que",
        "existem",
        "ha",
        "lista",
        "mostra"
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def get_all_school_documents() -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    results = collection.get(where={"type": "structured_school"})
    return results.get("documents", []), results.get("metadatas", [])


def get_all_course_documents() -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    results = collection.get(where={"type": "structured_course"})
    return results.get("documents", []), results.get("metadatas", [])


def get_school_documents_by_location(location: str) -> Tuple[List[str], List[dict]]:
    all_documents, all_metadatas = get_all_school_documents()

    filtered_documents = []
    filtered_metadatas = []

    for doc, meta in zip(all_documents, all_metadatas):
        meta_location = meta.get("local", "")

        if normalize_text(meta_location) == normalize_text(location):
            filtered_documents.append(doc)
            filtered_metadatas.append(meta)

    return filtered_documents, filtered_metadatas


def get_courses_filtered(
    degree: Optional[str] = None,
    school_code: Optional[str] = None,
    location: Optional[str] = None
) -> Tuple[List[str], List[dict]]:
    all_documents, all_metadatas = get_all_course_documents()

    filtered_documents = []
    filtered_metadatas = []

    for doc, meta in zip(all_documents, all_metadatas):
        meta_degree = meta.get("grau", "")
        meta_school = meta.get("escola", "")
        meta_location = meta.get("local", "")

        if degree and normalize_text(meta_degree) != normalize_text(degree):
            continue

        if school_code and normalize_text(meta_school) != normalize_text(school_code):
            continue

        if location and normalize_text(meta_location) != normalize_text(location):
            continue

        filtered_documents.append(doc)
        filtered_metadatas.append(meta)

    return filtered_documents, filtered_metadatas


def get_recommended_courses(question: str, n_results: int = 8) -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    embedding_model = get_embedding_model()

    expanded_question = expand_recommendation_query(question)
    query_embedding = embedding_model.encode([expanded_question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"type": "structured_course"}
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return documents, metadatas


def build_direct_school_answer(metadatas: List[dict], location: Optional[str] = None) -> str:
    schools = []

    for meta in metadatas:
        escola = meta.get("escola", "")
        sigla = meta.get("sigla", "")
        local = meta.get("local", "")

        if escola and sigla and local:
            schools.append((sigla, escola, local))

    schools = sorted(set(schools), key=lambda x: x[0])

    if not schools:
        return "Não encontrei essa informação nos documentos disponíveis."

    if location:
        lines = [f"Existem as seguintes escolas do IPVC em {location}:"]
    else:
        lines = ["Existem as seguintes escolas do IPVC:"]

    for sigla, escola, local in schools:
        lines.append(f"- {sigla} - {escola} ({local})")

    return "\n".join(lines)


def build_direct_courses_answer(
    metadatas: List[dict],
    degree: Optional[str] = None,
    school_code: Optional[str] = None,
    location: Optional[str] = None
) -> str:
    courses = []

    for meta in metadatas:
        curso = meta.get("curso", "")
        escola = meta.get("escola", "")
        grau = meta.get("grau", "")
        local = meta.get("local", "")
        regime = meta.get("regime", "")
        estado = meta.get("estado", "")
        observacoes = meta.get("observacoes", "")

        if not curso:
            continue

        parts = [curso]

        details = []

        if grau:
            details.append(grau)

        if escola:
            details.append(escola)

        if local:
            details.append(local)

        if regime:
            details.append(regime)

        if details:
            parts.append(f"({', '.join(details)})")

        extra = []

        if estado:
            extra.append(estado)

        if observacoes:
            extra.append(observacoes)

        line = " ".join(parts)

        if extra:
            line += f" — {'; '.join(extra)}"

        courses.append(line)

    courses = sorted(set(courses))

    if not courses:
        return "Não encontrei essa informação nos documentos disponíveis."

    header_parts = []

    if degree:
        header_parts.append(degree)

    if school_code:
        header_parts.append(f"na {school_code}")

    if location:
        header_parts.append(f"em {location}")

    if header_parts:
        header = " ".join(header_parts)
        lines = [f"Encontrei os seguintes cursos/formações de {header}:"]
    else:
        lines = ["Encontrei os seguintes cursos/formações no IPVC:"]

    for course in courses:
        lines.append(f"- {course}")

    return "\n".join(lines)


def search_relevant_context(question: str, n_results: int = 6) -> Tuple[List[str], List[dict]]:
    if is_school_question(question):
        location = detect_location(question)

        if location:
            return get_school_documents_by_location(location)

        return get_all_school_documents()

    if is_structured_courses_question(question):
        degree = detect_degree(question)
        school_code = detect_school_code(question)
        location = detect_location(question)

        return get_courses_filtered(
            degree=degree,
            school_code=school_code,
            location=location
        )

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


def build_recommendation_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)

    return f"""
És um assistente académico especializado na recomendação de cursos e formações do IPVC.

A tua tarefa é recomendar cursos com base nos interesses indicados pelo utilizador.

Regras obrigatórias:
- Usa apenas os cursos/formações presentes no contexto.
- Não inventes cursos.
- Não uses conhecimento externo.
- Recomenda no máximo 3 opções.
- Para cada opção, explica brevemente a relação com o interesse do utilizador.
- Indica o grau quando estiver disponível, por exemplo Licenciatura, Mestrado, Pós-graduação ou CTeSP.
- Indica a escola quando estiver disponível.
- Se nenhum curso estiver claramente relacionado com o interesse, responde apenas: "Não encontrei cursos relacionados com esse interesse nos documentos disponíveis."
- Responde em português de Portugal.
- Usa tópicos.

Contexto:
{context_text}

Pergunta do utilizador:
{question}

Resposta:
""".strip()


def ask_bot(question: str) -> Tuple[str, List[dict], List[str]]:
    if is_recommendation_question(question):
        context_chunks, metadatas = get_recommended_courses(question)

        prompt = build_recommendation_prompt(question, context_chunks)

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

    context_chunks, metadatas = search_relevant_context(question)

    if is_school_question(question):
        location = detect_location(question)
        answer = build_direct_school_answer(metadatas, location)
        return answer, metadatas, context_chunks

    if is_structured_courses_question(question):
        degree = detect_degree(question)
        school_code = detect_school_code(question)
        location = detect_location(question)

        answer = build_direct_courses_answer(
            metadatas=metadatas,
            degree=degree,
            school_code=school_code,
            location=location
        )

        return answer, metadatas, context_chunks

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