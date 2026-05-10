from typing import List, Tuple, Optional
from functools import lru_cache
import unicodedata
import re
from difflib import SequenceMatcher
import csv
import chromadb
import ollama
from sentence_transformers import SentenceTransformer


CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "ipvc_courses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3:latest"
FALLBACK_ANSWER = "Não encontrei essa informação nos documentos disponíveis."


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


def clean_portuguese_pt(text: str) -> str:
    replacements = {
        "você está interessado": "tens interesse",
        "você está interessada": "tens interesse",
        "se você está interessado": "se tens interesse",
        "se você está interessada": "se tens interesse",
        "você": "tu",
        "se concentra": "centra-se",
        "se concentram": "centram-se",
        "em uma": "numa",
        "em um": "num",
        "está procurando": "procuras",
        "relacionado à": "relacionado com a",
        "relacionada à": "relacionada com a",
        "tnuma": "tem uma",
        "tu está": "tu estás",
    }

    cleaned = str(text).strip()

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
        cleaned = cleaned.replace(old.capitalize(), new.capitalize())

    if FALLBACK_ANSWER in cleaned and cleaned != FALLBACK_ANSWER:
        return FALLBACK_ANSWER

    return cleaned.strip()


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
        "melgaco": "Melgaço",
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
        "tesp": "CTeSP",
    }

    for keyword, degree in degree_keywords.items():
        if normalize_text(keyword) in question_norm:
            return degree

    return None


def detect_phase(question: str) -> Optional[str]:
    question_norm = normalize_text(question)

    first_phase_patterns = [
        r"\b1\s*\.?\s*[ªa]?\s*fase\b",
        r"\bprimeira fase\b",
        r"\b1 fase\b",
    ]

    second_phase_patterns = [
        r"\b2\s*\.?\s*[ªa]?\s*fase\b",
        r"\bsegunda fase\b",
        r"\b2 fase\b",
    ]

    for pattern in first_phase_patterns:
        if re.search(pattern, question_norm):
            return "1.ª fase CNA"

    for pattern in second_phase_patterns:
        if re.search(pattern, question_norm):
            return "2.ª fase CNA"

    return None


def is_out_of_scope_question(question: str) -> bool:
    question_norm = normalize_text(question)

    external_terms = [
        "universidade do porto",
        "universidade de lisboa",
        "universidade de coimbra",
        "universidade do minho",
        "instituto superior tecnico",
        "politecnico do porto",
        "politecnico de braga",
        "estados unidos",
        "presidente dos estados unidos",
        "restaurante",
        "meteorologia",
        "tempo hoje",
        "futebol",
        "benfica",
        "porto futebol",
        "sporting",
        "cinema",
        "filme",
        "musica",
        "receita",
        "cozinhar",
    ]

    return any(normalize_text(term) in question_norm for term in external_terms)


def get_all_school_documents() -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    results = collection.get(where={"type": "structured_school"})
    return results.get("documents", []), results.get("metadatas", [])


def get_all_course_documents() -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    results = collection.get(where={"type": "structured_course"})
    return results.get("documents", []), results.get("metadatas", [])


def get_all_admission_documents() -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    results = collection.get(where={"type": "structured_admission"})
    return results.get("documents", []), results.get("metadatas", [])


def get_all_pdf_documents() -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    results = collection.get(where={"type": "pdf_chunk"})
    return results.get("documents", []), results.get("metadatas", [])


def get_meaningful_words(text: str) -> set:
    text_norm = normalize_text(text)

    stopwords = {
        "de", "da", "do", "das", "dos", "e", "em", "a", "o", "as", "os",
        "curso", "cursos", "licenciatura", "mestrado", "ctesp", "ipvc",
        "media", "nota", "ultimo", "colocado", "vagas", "quais", "qual",
        "para", "com", "uma", "um", "que", "como", "existem", "fase",
        "tenho", "gosto", "interesse", "recomendas", "recomenda", "aconselhas",
    }

    words = re.findall(r"\b[a-z0-9]+\b", text_norm)
    return {word for word in words if word not in stopwords and len(word) > 2}


def course_key(meta: dict) -> Tuple[str, str, str]:
    return (
        normalize_text(meta.get("curso", "")),
        normalize_text(meta.get("grau", "")),
        normalize_text(meta.get("escola", "")),
    )


def course_display_details(meta: dict, include_local: bool = True) -> str:
    details = []

    grau = meta.get("grau", "")
    escola = meta.get("escola", "")
    local = meta.get("local", "")

    if grau:
        details.append(str(grau))
    if escola:
        details.append(str(escola))
    if include_local and local:
        details.append(str(local))

    return ", ".join(details)


def find_best_course_match(question: str) -> Tuple[Optional[str], Optional[dict]]:
    documents, metadatas = get_all_course_documents()
    question_norm = normalize_text(question)
    question_words = get_meaningful_words(question)

    exact = []
    scored = []

    for doc, meta in zip(documents, metadatas):
        curso = meta.get("curso", "")
        if not curso:
            continue

        curso_norm = normalize_text(curso)

        if curso_norm and curso_norm in question_norm:
            exact.append((question_norm.find(curso_norm), -len(curso_norm), doc, meta))
            continue

        course_words = get_meaningful_words(curso)
        overlap = question_words.intersection(course_words)

        if overlap:
            score = len(overlap) / max(len(course_words), 1)
            if len(overlap) >= 2 or score >= 0.5:
                scored.append((score, len(overlap), doc, meta))

    if exact:
        exact = sorted(exact, key=lambda item: (item[0], item[1]))
        return exact[0][2], exact[0][3]

    if scored:
        scored = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2], scored[0][3]

    return None, None


def is_school_question(question: str) -> bool:
    question_norm = normalize_text(question)

    keywords = [
        "que escolas",
        "quais sao as escolas",
        "escolas do ipvc",
        "escolas existem",
        "quais as escolas",
        "lista de escolas",
        "mostrar escolas",
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
        "mostra",
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def get_school_documents_by_location(location: str) -> Tuple[List[str], List[dict]]:
    all_documents, all_metadatas = get_all_school_documents()

    filtered_documents = []
    filtered_metadatas = []

    for doc, meta in zip(all_documents, all_metadatas):
        if normalize_text(meta.get("local", "")) == normalize_text(location):
            filtered_documents.append(doc)
            filtered_metadatas.append(meta)

    return filtered_documents, filtered_metadatas


def get_courses_filtered(
    degree: Optional[str] = None,
    school_code: Optional[str] = None,
    location: Optional[str] = None,
) -> Tuple[List[str], List[dict]]:
    all_documents, all_metadatas = get_all_course_documents()

    filtered_documents = []
    filtered_metadatas = []

    for doc, meta in zip(all_documents, all_metadatas):
        if degree and normalize_text(meta.get("grau", "")) != normalize_text(degree):
            continue

        if school_code and normalize_text(meta.get("escola", "")) != normalize_text(school_code):
            continue

        if location and normalize_text(meta.get("local", "")) != normalize_text(location):
            continue

        filtered_documents.append(doc)
        filtered_metadatas.append(meta)

    return filtered_documents, filtered_metadatas


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
        return FALLBACK_ANSWER

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
    location: Optional[str] = None,
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
        return FALLBACK_ANSWER

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


def is_admission_question(question: str) -> bool:
    question_norm = normalize_text(question)

    keywords = [
        "media",
        "medias",
        "nota",
        "notas",
        "ultimo colocado",
        "ultima colocada",
        "ultimo colocados",
        "vagas",
        "colocados",
        "candidatura",
        "acesso",
        "dges",
        "sobras",
        "1 fase",
        "1.ª fase",
        "primeira fase",
        "2 fase",
        "2.ª fase",
        "segunda fase",
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def parse_number(value) -> Optional[float]:
    if value is None:
        return None

    value_str = str(value).strip().replace(",", ".")

    if not value_str or value_str.lower() in {"nan", "none", "null"}:
        return None

    try:
        return float(value_str)
    except ValueError:
        return None


def normalize_grade_to_200(value: float) -> float:
    if value <= 20:
        return value * 10

    return value


def format_grade(value) -> str:
    number = parse_number(value)

    if number is None:
        return "não disponível"

    if number > 20:
        return f"{number:.1f} pontos (equivalente a {number / 10:.2f}/20)"

    return f"{number:.2f}/20"


def detect_average_threshold(question: str) -> Optional[float]:
    question_norm = normalize_text(question)

    patterns = [
        r"(?:abaixo|inferior|menor|ate|até)\s+(?:de|a)?\s*(\d+(?:[.,]\d+)?)",
        r"tenho\s+(?:media|nota)\s+(?:de)?\s*(\d+(?:[.,]\d+)?)",
        r"com\s+(?:media|nota)\s+(?:de)?\s*(\d+(?:[.,]\d+)?)",
        r"(?:media|nota)\s+(?:abaixo|inferior|menor|ate|até)\s+(?:de|a)?\s*(\d+(?:[.,]\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question_norm)

        if match:
            number = parse_number(match.group(1))

            if number is not None:
                return normalize_grade_to_200(number)

    return None


def match_admission_courses(question: str, pairs: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    question_norm = normalize_text(question)
    question_words = get_meaningful_words(question)

    scored_pairs = []

    for doc, meta in pairs:
        course = meta.get("curso", "")
        course_norm = normalize_text(course)
        course_words = get_meaningful_words(course)

        if not course:
            continue

        if course_norm and course_norm in question_norm:
            scored_pairs.append((100, doc, meta))
            continue

        if not course_words:
            continue

        overlap = question_words.intersection(course_words)

        if len(course_words) == 1:
            score = len(overlap)
        else:
            score = len(overlap) / len(course_words)

        if len(overlap) >= 2 or score >= 0.6:
            scored_pairs.append((score, doc, meta))

    scored_pairs = sorted(scored_pairs, key=lambda item: item[0], reverse=True)

    if not scored_pairs:
        return []

    best_score = scored_pairs[0][0]

    return [
        (doc, meta)
        for score, doc, meta in scored_pairs
        if score == best_score or score >= 0.6
    ]


def filter_admission_pairs(question: str, pairs: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    school_code = detect_school_code(question)
    degree = detect_degree(question)
    phase = detect_phase(question)

    filtered = []

    for doc, meta in pairs:
        if school_code and normalize_text(meta.get("escola", "")) != normalize_text(school_code):
            continue

        if degree and normalize_text(meta.get("grau", "")) != normalize_text(degree):
            continue

        if phase and normalize_text(phase) != normalize_text(meta.get("fase", "")):
            continue

        filtered.append((doc, meta))

    return filtered


def build_admission_line(meta: dict) -> str:
    curso = meta.get("curso", "")
    escola = meta.get("escola", "")
    ano = meta.get("ano", "")
    fase = meta.get("fase", "")
    vagas = meta.get("vagas_iniciais", "")
    colocados = meta.get("colocados", "")
    nota = meta.get("nota_ultimo_colocado_contingente_geral", "")
    sobras = meta.get("sobras_2_fase", "")

    header_details = []

    if escola:
        header_details.append(str(escola))
    if ano:
        header_details.append(str(ano))
    if fase:
        header_details.append(str(fase))

    if header_details:
        line = f"- {curso} ({', '.join(header_details)})"
    else:
        line = f"- {curso}"

    line += f"\n• Nota do último colocado: {format_grade(nota)}"

    if vagas != "":
        line += f"\n• Vagas iniciais: {vagas}"
    if colocados != "":
        line += f"\n• Colocados: {colocados}"
    if sobras != "":
        line += f"\n• Sobras para a 2.ª fase: {sobras}\n"

    return line


def answer_admission_question(question: str) -> Tuple[str, List[dict], List[str]]:
    phase = detect_phase(question)

    documents, metadatas = get_all_admission_documents()
    pairs = list(zip(documents, metadatas))

    if not pairs:
        return FALLBACK_ANSWER, [], []

    pairs = filter_admission_pairs(question, pairs)

    if not pairs:
        return FALLBACK_ANSWER, [], []

    threshold = detect_average_threshold(question)

    if threshold is not None:
        valid_pairs = []

        for doc, meta in pairs:
            grade = parse_number(meta.get("nota_ultimo_colocado_contingente_geral", ""))

            if grade is None:
                continue

            grade_200 = normalize_grade_to_200(grade)

            if grade_200 <= threshold:
                valid_pairs.append((doc, meta, grade_200))

        valid_pairs = sorted(valid_pairs, key=lambda item: item[2], reverse=True)

        if not valid_pairs:
            return FALLBACK_ANSWER, [], []

        selected = valid_pairs[:12]

        if phase:
            phase_text = phase
        else:
            phase_text = "nas fases disponíveis do CNA 2025"

        lines = [
            f"Com uma média até {threshold / 10:.2f}/20, encontrei estes cursos do IPVC com nota do último colocado igual ou inferior {phase_text}:"
        ]

        for doc, meta, _ in selected:
            lines.append(build_admission_line(meta))

        selected_docs = [doc for doc, _, _ in selected]
        selected_metas = [meta for _, meta, _ in selected]

        return "\n".join(lines), selected_metas, selected_docs

    matched_pairs = match_admission_courses(question, pairs)

    if matched_pairs:
        lines = ["Encontrei os seguintes dados de acesso:"]

        for doc, meta in matched_pairs[:6]:
            lines.append(build_admission_line(meta))

        selected_docs = [doc for doc, _ in matched_pairs[:6]]
        selected_metas = [meta for _, meta in matched_pairs[:6]]

        return "\n".join(lines), selected_metas, selected_docs

    school_code = detect_school_code(question)
    degree = detect_degree(question)

    if school_code or degree:
        lines = ["Encontrei os seguintes dados de acesso:"]

        for doc, meta in pairs[:15]:
            lines.append(build_admission_line(meta))

        selected_docs = [doc for doc, _ in pairs[:15]]
        selected_metas = [meta for _, meta in pairs[:15]]

        return "\n".join(lines), selected_metas, selected_docs

    return FALLBACK_ANSWER, [], []


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
        "o que devo escolher",
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def is_similar_word(word: str, target: str, threshold: float = 0.84) -> bool:
    word_norm = normalize_text(word)
    target_norm = normalize_text(target)

    if not word_norm or not target_norm:
        return False

    if word_norm == target_norm:
        return True

    if len(word_norm) < 5 or len(target_norm) < 5:
        return False

    return SequenceMatcher(None, word_norm, target_norm).ratio() >= threshold


def detect_interest_terms(question: str) -> List[str]:
    question_norm = normalize_text(question)
    question_words = re.findall(r"\b[a-z0-9]+\b", question_norm)

    interest_map = {
        "animais": ["animais", "animal", "veterinaria", "saude animal", "cuidados", "clinica"],
        "animal": ["animais", "animal", "veterinaria", "saude animal", "cuidados", "clinica"],
        "veterinaria": ["animais", "animal", "veterinaria", "saude animal", "cuidados", "clinica"],
        "programacao": ["programacao", "programar", "software", "informatica", "computadores", "tecnologia", "redes", "sistemas", "inteligencia artificial"],
        "programar": ["programacao", "programar", "software", "informatica", "computadores", "tecnologia", "redes", "sistemas", "inteligencia artificial"],
        "software": ["programacao", "software", "informatica", "computadores", "tecnologia", "aplicacoes"],
        "computadores": ["informatica", "computadores", "redes", "sistemas", "programacao", "software"],
        "informatica": ["informatica", "programacao", "software", "redes", "sistemas", "computadores"],
        "redes": ["redes", "sistemas", "computadores", "ciberseguranca", "servidores", "infraestruturas"],
        "seguranca": ["ciberseguranca", "seguranca", "redes", "sistemas", "informatica"],
        "saude": ["saude", "enfermagem", "cuidados", "fisioterapia", "gerontologia"],
        "enfermagem": ["saude", "enfermagem", "cuidados", "hospitalar", "comunitaria"],
        "gestao": ["gestao", "empresas", "administracao", "contabilidade", "marketing", "negocios"],
        "empresas": ["gestao", "empresas", "administracao", "contabilidade", "marketing", "negocios"],
        "contabilidade": ["contabilidade", "fiscalidade", "financas", "auditoria", "empresas"],
        "fiscalidade": ["contabilidade", "fiscalidade", "financas", "auditoria", "empresas"],
        "marketing": ["marketing", "comunicacao", "vendas", "gestao comercial"],
        "desporto": ["desporto", "atividade fisica", "treino", "exercicio", "bem-estar"],
        "educacao": ["educacao", "ensino", "criancas", "formacao", "intervencao educativa"],
        "criancas": ["educacao", "criancas", "infancia", "ensino", "intervencao educativa"],
        "ambiente": ["ambiente", "sustentabilidade", "agricultura", "recursos naturais", "agronomia"],
        "agricultura": ["agricultura", "agronomia", "ambiente", "sustentabilidade", "recursos naturais"],
        "turismo": ["turismo", "hotelaria", "gestao turistica", "patrimonio", "lazer"],
        "design": ["design", "multimedia", "criatividade", "comunicacao visual", "produto digital"],
        "jogos": ["jogos", "videojogos", "multimedia", "computacao grafica", "animacao", "programacao"],
        "videojogos": ["jogos", "videojogos", "multimedia", "computacao grafica", "animacao", "programacao"],
        "matematica": [
            "matematica", "raciocinio logico", "logica", "informatica", "programacao",
            "software", "redes", "sistemas", "engenharia", "computadores", "tecnologia",
            "estatistica", "dados", "financas", "contabilidade"
        ],
        "matemática": [
            "matematica", "raciocinio logico", "logica", "informatica", "programacao",
            "software", "redes", "sistemas", "engenharia", "computadores", "tecnologia",
            "estatistica", "dados", "financas", "contabilidade"
        ],
        "numeros": ["matematica", "estatistica", "dados", "financas", "contabilidade", "gestao", "engenharia"],
        "números": ["matematica", "estatistica", "dados", "financas", "contabilidade", "gestao", "engenharia"],
        "calculos": ["matematica", "engenharia", "informatica", "financas", "contabilidade"],
        "cálculos": ["matematica", "engenharia", "informatica", "financas", "contabilidade"],
        "estatistica": ["estatistica", "dados", "matematica", "informatica", "gestao", "financas"],
        "estatística": ["estatistica", "dados", "matematica", "informatica", "gestao", "financas"],
        "dados": ["dados", "estatistica", "informatica", "programacao", "software", "gestao"],
        "logica": ["logica", "raciocinio logico", "matematica", "programacao", "informatica", "engenharia"],
        "lógica": ["logica", "raciocinio logico", "matematica", "programacao", "informatica", "engenharia"],
    }

    terms = []

    for keyword, expansion_terms in interest_map.items():
        keyword_norm = normalize_text(keyword)

        found_exact = keyword_norm in question_norm
        found_fuzzy = any(is_similar_word(word, keyword_norm) for word in question_words)

        if found_exact or found_fuzzy:
            terms.extend(expansion_terms)

    # Mantém algumas palavras relevantes da pergunta, mas sem deixar que erros
    # ortográficos dominem a recomendação.
    for word in get_meaningful_words(question):
        if len(word) >= 4 and word not in terms:
            terms.append(word)

    seen = set()
    unique_terms = []

    for term in terms:
        term_norm = normalize_text(term)
        if term_norm and term_norm not in seen:
            seen.add(term_norm)
            unique_terms.append(term_norm)

    return unique_terms

def course_searchable_text(meta: dict) -> str:
    return " ".join([
        str(meta.get("curso", "")),
        str(meta.get("area", "")),
        str(meta.get("resumo", "")),
        str(meta.get("descricao", "")),
        str(meta.get("interesses_relacionados", "")),
        str(meta.get("palavras_chave", "")),
        str(meta.get("saidas_profissionais", "")),
        str(meta.get("provas_ingresso", "")),
        str(meta.get("provas_ingresso_tags", "")),
    ])


def score_course_for_interest(meta: dict, question: str) -> int:
    terms = detect_interest_terms(question)
    if not terms:
        return 0

    full_text = normalize_text(course_searchable_text(meta))
    high_weight_text = normalize_text(" ".join([
        str(meta.get("curso", "")),
        str(meta.get("area", "")),
        str(meta.get("interesses_relacionados", "")),
        str(meta.get("palavras_chave", "")),
        str(meta.get("provas_ingresso", "")),
        str(meta.get("provas_ingresso_tags", "")),
    ]))

    score = 0

    for term in terms:
        if not term:
            continue

        if term in high_weight_text:
            score += 3
        elif term in full_text:
            score += 1

    return score


def get_recommended_courses(question: str, n_results: int = 8) -> Tuple[List[str], List[dict]]:
    all_documents, all_metadatas = get_all_course_documents()

    scored = []
    used = set()

    for doc, meta in zip(all_documents, all_metadatas):
        key = course_key(meta)
        if key in used:
            continue
        used.add(key)

        score = score_course_for_interest(meta, question)
        if score > 0:
            scored.append((score, doc, meta))

    if scored:
        scored = sorted(scored, key=lambda item: item[0], reverse=True)
        selected = scored[:n_results]
        return [doc for _, doc, _ in selected], [meta for _, _, meta in selected]

    collection = get_collection()
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"type": "structured_course"},
    )

    return results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]


def find_best_admission_for_course(course_meta: dict, preferred_phase: Optional[str] = None) -> Optional[dict]:
    _, admission_metas = get_all_admission_documents()

    course_name = normalize_text(course_meta.get("curso", ""))
    school = normalize_text(course_meta.get("escola", ""))
    degree = normalize_text(course_meta.get("grau", ""))

    candidates = []

    for meta in admission_metas:
        if normalize_text(meta.get("curso", "")) != course_name:
            continue

        if school and normalize_text(meta.get("escola", "")) != school:
            continue

        if degree and normalize_text(meta.get("grau", "")) != degree:
            continue

        if preferred_phase and normalize_text(meta.get("fase", "")) != normalize_text(preferred_phase):
            continue

        candidates.append(meta)

    if not candidates and preferred_phase:
        return find_best_admission_for_course(course_meta, preferred_phase=None)

    if not candidates:
        return None

    def sort_key(meta: dict):
        grade = parse_number(meta.get("nota_ultimo_colocado_contingente_geral", ""))
        grade_200 = normalize_grade_to_200(grade) if grade is not None else 9999
        phase = normalize_text(meta.get("fase", ""))
        phase_priority = 0 if "1." in phase or "1ª" in phase or "1a" in phase else 1
        grade_available = 0 if grade is not None else 1
        return grade_available, grade_200, phase_priority

    return sorted(candidates, key=sort_key)[0]


def build_recommendation_answer(question: str, course_docs: List[str], course_metas: List[dict]) -> Tuple[str, List[dict], List[str]]:
    if not course_metas:
        return FALLBACK_ANSWER, [], []

    threshold = detect_average_threshold(question)
    preferred_phase = detect_phase(question)

    enriched = []
    used_keys = set()

    for doc, meta in zip(course_docs, course_metas):
        if not meta.get("curso", ""):
            continue

        key = course_key(meta)
        if key in used_keys:
            continue
        used_keys.add(key)

        admission = find_best_admission_for_course(meta, preferred_phase=preferred_phase)
        grade = None

        if admission:
            grade = parse_number(admission.get("nota_ultimo_colocado_contingente_geral", ""))
            if grade is not None:
                grade = normalize_grade_to_200(grade)

        interest_score = score_course_for_interest(meta, question)

        if threshold is None:
            score_group = 0
        elif grade is None:
            score_group = 1
        elif grade <= threshold:
            score_group = 0
        else:
            score_group = 2

        distance = abs(grade - threshold) if grade is not None and threshold is not None else 0

        enriched.append({
            "doc": doc,
            "course_meta": meta,
            "admission_meta": admission,
            "grade": grade,
            "score_group": score_group,
            "distance": distance,
            "interest_score": interest_score,
        })

    if threshold is not None:
        enriched = sorted(enriched, key=lambda item: (item["score_group"], -item["interest_score"], item["distance"]))
    else:
        enriched = sorted(enriched, key=lambda item: (-item["interest_score"], normalize_text(item["course_meta"].get("curso", ""))))

    selected = enriched[:3]

    if not selected:
        return FALLBACK_ANSWER, [], []

    if threshold is not None:
        lines = [f"Com base no teu interesse e na média indicada ({threshold / 10:.2f}/20), recomendo:"]
    else:
        lines = ["Com base no teu interesse, recomendo:"]

    returned_metas = []
    returned_docs = []

    for item in selected:
        meta = item["course_meta"]
        admission = item["admission_meta"]
        grade = item["grade"]

        curso = meta.get("curso", "")
        resumo = meta.get("resumo", "") or meta.get("descricao", "")
        area = meta.get("area", "")
        details = course_display_details(meta, include_local=True)

        if resumo:
            explanation = resumo
        elif area:
            explanation = f"é uma opção relacionada com a área de {area}"
        else:
            explanation = "é uma opção relacionada com o interesse indicado"

        line = f"- {curso}"

        if details:
            line += f" ({details})"

        line += f" — {explanation}"

        provas = str(meta.get("provas_ingresso", "")).strip()
        if provas:
            line += f" Provas de ingresso: {provas}."

        if admission and grade is not None:
            fase = admission.get("fase", "")
            fase_text = f" na {fase}" if fase else ""
            line += f" Nota do último colocado{fase_text}: {format_grade(admission.get('nota_ultimo_colocado_contingente_geral', ''))}."

            if threshold is not None:
                if grade <= threshold:
                    line += " A tua média fica dentro da nota indicada nos dados disponíveis."
                else:
                    line += " A tua média fica abaixo da nota indicada nos dados disponíveis, por isso pode ser uma opção mais difícil."
        elif threshold is not None:
            line += " Não encontrei nota do último colocado disponível para confirmar a compatibilidade com a tua média."

        lines.append(line)
        returned_metas.append(meta)
        returned_docs.append(item["doc"])

        if admission:
            returned_metas.append(admission)
            returned_docs.append(str(admission))

    return "\n".join(lines), returned_metas, returned_docs


def is_career_question(question: str) -> bool:
    question_norm = normalize_text(question)

    keywords = [
        "saidas profissionais",
        "saidas",
        "saída profissional",
        "saídas profissionais",
        "profissoes",
        "profissões",
        "emprego",
        "empregabilidade",
        "mercado de trabalho",
        "trabalhar em",
        "trabalhar na area",
        "trabalhar na área",
        "trabalhar com",
        "carreira",
        "o que posso fazer",
        "onde posso trabalhar",
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def match_courses_for_career_question(question: str) -> Tuple[List[str], List[dict]]:
    all_documents, all_metadatas = get_all_course_documents()
    question_norm = normalize_text(question)
    question_words = set(re.findall(r"\b[a-z0-9]+\b", question_norm))

    scored = []

    for doc, meta in zip(all_documents, all_metadatas):
        curso = meta.get("curso", "")
        if not curso:
            continue

        searchable_norm = normalize_text(course_searchable_text(meta))
        course_norm = normalize_text(curso)
        score = 0

        if course_norm and course_norm in question_norm:
            score += 100

        for word in question_words:
            if len(word) > 3 and word in searchable_norm:
                score += 1

        if score > 0:
            scored.append((score, doc, meta))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    selected = scored[:6]

    return [doc for _, doc, _ in selected], [meta for _, _, meta in selected]


def build_career_answer(question: str, metadatas: List[dict]) -> str:
    if not metadatas:
        return FALLBACK_ANSWER

    question_norm = normalize_text(question)
    specific_course = None

    for meta in metadatas:
        curso = meta.get("curso", "")
        if curso and normalize_text(curso) in question_norm:
            specific_course = meta
            break

    if specific_course:
        curso = specific_course.get("curso", "")
        details = course_display_details(specific_course, include_local=False)
        saidas = specific_course.get("saidas_profissionais", "")

        if not saidas:
            return FALLBACK_ANSWER

        lines = [f"As saídas profissionais de {curso} são:"]

        if details:
            lines.append(f"({details})")

        for item in str(saidas).split(";"):
            item = item.strip()
            if item:
                lines.append(f"- {item}")

        return "\n".join(lines)

    lines = ["Encontrei os seguintes cursos relacionados com essa área profissional:"]
    used = set()

    for meta in metadatas[:5]:
        curso = meta.get("curso", "")
        if not curso or curso in used:
            continue

        used.add(curso)
        details = course_display_details(meta, include_local=False)
        saidas = meta.get("saidas_profissionais", "")
        area = meta.get("area", "")

        info = curso
        if details:
            info += f" ({details})"

        if saidas:
            info += f" — saídas profissionais: {saidas}"
        elif area:
            info += f" — relacionado com a área de {area}"

        lines.append(f"- {info}")

    if len(lines) == 1:
        return FALLBACK_ANSWER

    return "\n".join(lines)


def is_comparison_question(question: str) -> bool:
    question_norm = normalize_text(question)

    keywords = [
        "compara",
        "comparar",
        "comparacao",
        "comparação",
        "diferença",
        "diferenca",
        "diferenças",
        "diferencas",
        "qual e melhor",
        "qual é melhor",
        "qual e mais indicado",
        "qual é mais indicado",
        "qual devo escolher",
        "entre",
    ]

    if any(normalize_text(keyword) in question_norm for keyword in keywords):
        return True

    if " ou " in f" {question_norm} ":
        _, matched_metas = find_courses_for_comparison(question)
        return len(matched_metas) >= 2

    return False


def parse_comparison_course_phrases(question: str) -> List[str]:
    question_clean = re.sub(r"[?.!]+$", "", question.strip())
    question_clean = re.sub(r"\s+", " ", question_clean)

    patterns = [
        r"^(?:compara|comparar|compara-me|compara me)\s+(.+?)\s+com\s+(.+)$",
        r"^(?:qual(?:\s+é|\s+e)?\s+a\s+diferença\s+entre|qual(?:\s+é|\s+e)?\s+a\s+diferenca\s+entre|diferenças\s+entre|diferencas\s+entre|entre)\s+(.+?)\s+e\s+(.+)$",
        r"^(.+?)\s+ou\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, question_clean, flags=re.IGNORECASE)
        if match:
            return [part.strip(" ,.;:") for part in match.groups() if part.strip(" ,.;:")]

    return []


def score_course_name_against_phrase(course_name: str, phrase: str) -> float:
    course_norm = normalize_text(course_name)
    phrase_norm = normalize_text(phrase)

    if not course_norm or not phrase_norm:
        return 0

    if course_norm == phrase_norm:
        return 100

    if course_norm in phrase_norm:
        return 95 + min(len(course_norm), 40) / 100

    if phrase_norm in course_norm:
        return 90 - max(0, len(course_norm) - len(phrase_norm)) / 100

    phrase_words = get_meaningful_words(phrase_norm)
    course_words = get_meaningful_words(course_norm)

    if not phrase_words or not course_words:
        return 0

    overlap = phrase_words.intersection(course_words)

    if not overlap:
        fuzzy_overlap = 0
        for pw in phrase_words:
            if any(is_similar_word(pw, cw, threshold=0.86) for cw in course_words):
                fuzzy_overlap += 1
        if fuzzy_overlap == 0:
            return 0

        return 55 + (fuzzy_overlap / len(phrase_words)) * 20

    coverage = len(overlap) / len(phrase_words)
    precision = len(overlap) / len(course_words)

    return 60 + coverage * 25 + precision * 10


def find_best_course_for_phrase(
    phrase: str,
    documents: List[str],
    metadatas: List[dict],
    excluded_keys: Optional[set] = None
) -> Tuple[Optional[str], Optional[dict]]:
    excluded_keys = excluded_keys or set()

    scored = []

    for doc, meta in zip(documents, metadatas):
        curso = meta.get("curso", "")

        if not curso:
            continue

        key = course_key(meta)
        if key in excluded_keys:
            continue

        score = score_course_name_against_phrase(curso, phrase)

        if score > 0:
            scored.append((score, -len(normalize_text(curso)), doc, meta))

    if not scored:
        return None, None

    scored = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2], scored[0][3]


def find_courses_for_comparison(question: str, max_courses: int = 3) -> Tuple[List[str], List[dict]]:
    all_documents, all_metadatas = get_all_course_documents()

    phrases = parse_comparison_course_phrases(question)
    selected_docs = []
    selected_metas = []
    used_keys = set()

    if len(phrases) >= 2:
        for phrase in phrases[:max_courses]:
            doc, meta = find_best_course_for_phrase(
                phrase=phrase,
                documents=all_documents,
                metadatas=all_metadatas,
                excluded_keys=used_keys,
            )

            if meta:
                selected_docs.append(doc or "")
                selected_metas.append(meta)
                used_keys.add(course_key(meta))

        if len(selected_metas) >= 2:
            return selected_docs, selected_metas

    question_norm = normalize_text(question)
    exact_matches = []

    for doc, meta in zip(all_documents, all_metadatas):
        curso = meta.get("curso", "")
        if not curso:
            continue

        curso_norm = normalize_text(curso)
        if curso_norm and curso_norm in question_norm:
            key = course_key(meta)
            if key not in used_keys:
                used_keys.add(key)
                exact_matches.append((question_norm.find(curso_norm), -len(curso_norm), doc, meta))

    exact_matches = sorted(exact_matches, key=lambda item: (item[0], item[1]))

    if len(exact_matches) >= 2:
        selected = exact_matches[:max_courses]
        return [doc for _, _, doc, _ in selected], [meta for _, _, _, meta in selected]

    question_words = get_meaningful_words(question)
    scored = []

    for doc, meta in zip(all_documents, all_metadatas):
        curso = meta.get("curso", "")
        if not curso:
            continue

        key = course_key(meta)
        if key in used_keys:
            continue

        course_words = get_meaningful_words(curso)
        overlap = question_words.intersection(course_words)

        if not overlap:
            continue

        score = len(overlap)
        scored.append((score, len(normalize_text(curso)), doc, meta))

    scored = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)

    selected_docs = []
    selected_metas = []

    if exact_matches:
        selected_docs.append(exact_matches[0][2])
        selected_metas.append(exact_matches[0][3])
        used_keys.add(course_key(exact_matches[0][3]))

    for _, _, doc, meta in scored:
        key = course_key(meta)

        if key in used_keys:
            continue

        selected_docs.append(doc)
        selected_metas.append(meta)
        used_keys.add(key)

        if len(selected_metas) >= max_courses:
            break

    return selected_docs[:max_courses], selected_metas[:max_courses]

def split_semicolon_items(text: str, max_items: int = 3) -> List[str]:
    items = []

    for item in str(text).split(";"):
        item = item.strip()
        if item:
            items.append(item)

        if len(items) >= max_items:
            break

    return items


def compact_focus(meta: dict) -> str:
    resumo = str(meta.get("resumo", "")).strip()
    area = str(meta.get("area", "")).strip()
    interesses = split_semicolon_items(meta.get("interesses_relacionados", ""), max_items=3)
    palavras = split_semicolon_items(meta.get("palavras_chave", ""), max_items=3)

    if resumo:
        return resumo

    if interesses:
        return "Formação relacionada com " + ", ".join(interesses) + "."

    if palavras:
        return "Formação relacionada com " + ", ".join(palavras) + "."

    if area:
        return f"Formação na área de {area}."

    return "Informação de síntese não disponível nos dados estruturados."


def build_comparison_difference_lines(meta1: dict, meta2: dict) -> List[str]:
    curso1 = meta1.get("curso", "Curso 1")
    curso2 = meta2.get("curso", "Curso 2")
    area1 = meta1.get("area", "")
    area2 = meta2.get("area", "")
    interests1 = split_semicolon_items(meta1.get("interesses_relacionados", ""), max_items=4)
    interests2 = split_semicolon_items(meta2.get("interesses_relacionados", ""), max_items=4)
    saidas1 = split_semicolon_items(meta1.get("saidas_profissionais", ""), max_items=3)
    saidas2 = split_semicolon_items(meta2.get("saidas_profissionais", ""), max_items=3)

    lines = []

    if area1 and area2 and normalize_text(area1) != normalize_text(area2):
        lines.append(f"{curso1} enquadra-se mais em {area1}, enquanto {curso2} se enquadra mais em {area2}.")
    elif area1 and area2:
        lines.append(f"Ambos pertencem à área de {area1}, mas têm focos diferentes dentro dessa área.")

    if interests1 or interests2:
        part1 = ", ".join(interests1) if interests1 else "os temas principais indicados no curso"
        part2 = ", ".join(interests2) if interests2 else "os temas principais indicados no curso"
        lines.append(f"{curso1} está mais associado a {part1}; {curso2} está mais associado a {part2}.")

    if saidas1 or saidas2:
        part1 = ", ".join(saidas1) if saidas1 else "saídas profissionais não especificadas"
        part2 = ", ".join(saidas2) if saidas2 else "saídas profissionais não especificadas"
        lines.append(f"Nas saídas profissionais, {curso1} aponta para {part1}; {curso2} aponta para {part2}.")

    if len(lines) < 3:
        lines.append("A escolha deve depender sobretudo da área técnica em que se pretende aprofundar competências.")

    return lines[:3]


def build_comparison_answer(metadatas: List[dict]) -> str:
    if len(metadatas) < 2:
        return FALLBACK_ANSWER

    meta1 = metadatas[0]
    meta2 = metadatas[1]

    curso1 = meta1.get("curso", "")
    curso2 = meta2.get("curso", "")

    lines = ["Comparação entre os cursos:", ""]

    for idx, meta in enumerate([meta1, meta2], start=1):
        curso = meta.get("curso", "")
        grau = meta.get("grau", "não disponível")
        escola = meta.get("escola", "não disponível")
        sintese = compact_focus(meta)

        lines.append(f"- Curso {idx}: {curso}")
        lines.append(f"Grau: {grau}")
        lines.append(f"Escola: {escola}")
        lines.append(f"Síntese: {sintese}")
        lines.append("")

    lines.append("Principais diferenças:")
    for difference in build_comparison_difference_lines(meta1, meta2):
        lines.append(f"- {difference}")

    focus1 = split_semicolon_items(meta1.get("interesses_relacionados", ""), max_items=3)
    focus2 = split_semicolon_items(meta2.get("interesses_relacionados", ""), max_items=3)

    focus1_text = ", ".join(focus1) if focus1 else meta1.get("area", "esta área")
    focus2_text = ", ".join(focus2) if focus2 else meta2.get("area", "esta área")

    lines.append("")
    lines.append("Indicação final:")
    lines.append(f"- {curso1} pode fazer mais sentido se procuras uma formação ligada a {focus1_text}.")
    lines.append(f"- {curso2} pode fazer mais sentido se procuras uma formação ligada a {focus2_text}.")

    return "\n".join(lines)


def answer_comparison_question(question: str) -> Tuple[str, List[dict], List[str]]:
    context_chunks, metadatas = find_courses_for_comparison(question)

    if len(metadatas) < 2:
        return FALLBACK_ANSWER, [], []

    answer = build_comparison_answer(metadatas)
    return answer, metadatas[:2], context_chunks[:2]


def is_entry_exam_question(question: str) -> bool:
    question_norm = normalize_text(question)

    keywords = [
        "provas de ingresso",
        "provas ingresso",
        "prova de ingresso",
        "exames nacionais",
        "exame nacional",
        "que exames preciso",
        "quais os exames",
        "exames para entrar",
        "exames preciso para entrar",
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def clean_exam_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name)).strip(" .;,:-/")

    stop_markers = [
        " Curso", " Cursos", " Escola", " Grau", " Duração", " Duracao", " Regime",
        " Vagas", " Candidatura", " Código", " Codigo", " Fonte", " Provas",
        " Local", " Condições", " Condicoes", " Observações", " Observacoes",
    ]

    for marker in stop_markers:
        index = name.find(marker)
        if index > 0:
            name = name[:index].strip(" .;,:-/")

    return name


def extract_entry_exams_from_text(text: str) -> List[str]:
    exams = []
    seen = set()

    code_pattern = re.compile(
        r"\[\s*(\d{1,2})\s*\]\s*([A-Za-zÀ-ÿ]+(?:\s+(?:[A-Za-zÀ-ÿ]+|e|de|da|do|das|dos|às|a|ao|à)){0,7})"
    )

    for match in code_pattern.finditer(text):
        code = match.group(1).zfill(2)
        name = clean_exam_name(match.group(2))

        if not name or len(name) < 3:
            continue

        exam = f"[{code}] {name}"
        key = normalize_text(exam)

        if key not in seen:
            seen.add(key)
            exams.append(exam)

    if exams:
        return exams

    text_norm = normalize_text(text)
    marker = "provas de ingresso"
    index = text_norm.find(marker)

    if index == -1:
        return []

    snippet = text[index:index + 350]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    snippet = re.sub(r"(?i)^provas?\s+de\s+ingresso\s*[:\-]?\s*", "", snippet).strip()

    for delimiter in [" Curso", " Escola", " Grau", " Duração", " Duracao", " Regime", " Vagas", " Fonte"]:
        pos = snippet.find(delimiter)
        if pos > 0:
            snippet = snippet[:pos].strip()

    if snippet:
        return [snippet.strip(" .;:")]

    return []


def get_entry_exams_from_course_metadata(meta: Optional[dict]) -> List[str]:
    if not meta:
        return []

    possible_keys = [
        "provas_ingresso",
        "provas_de_ingresso",
        "exames_ingresso",
        "exames_nacionais",
        "prova_ingresso",
    ]

    raw_values = []

    for key in possible_keys:
        value = str(meta.get(key, "")).strip()
        if value:
            raw_values.append(value)

    exams = []
    seen = set()

    for raw in raw_values:
        parts = re.split(r";|\n|\|", raw)

        for part in parts:
            item = part.strip(" -•\t")
            if not item:
                continue

            key = normalize_text(item)
            if key not in seen:
                seen.add(key)
                exams.append(item)

    return exams


def find_best_course_match(question: str) -> Tuple[Optional[str], Optional[dict]]:
    all_documents, all_metadatas = get_all_course_documents()
    question_norm = normalize_text(question)
    question_words = get_meaningful_words(question)

    best_score = 0
    best_doc = None
    best_meta = None

    for doc, meta in zip(all_documents, all_metadatas):
        curso = meta.get("curso", "")

        if not curso:
            continue

        curso_norm = normalize_text(curso)
        curso_words = get_meaningful_words(curso)

        score = 0

        if curso_norm and curso_norm in question_norm:
            score += 100

        overlap = question_words.intersection(curso_words)
        score += len(overlap) * 10

        if score > best_score:
            best_score = score
            best_doc = doc
            best_meta = meta

    if best_score <= 0:
        return None, None

    return best_doc, best_meta


def answer_entry_exam_question(question: str) -> Tuple[str, List[dict], List[str]]:
    course_doc, course_meta = find_best_course_match(question)
    course_name = course_meta.get("curso", "") if course_meta else ""

    # 1. Primeiro tenta responder pelos metadados da ChromaDB
    metadata_exams = get_entry_exams_from_course_metadata(course_meta)

    if metadata_exams:
        lines = [f"As provas de ingresso para {course_name} são:"]

        for exam in metadata_exams:
            lines.append(f"- {exam}")

        fonte = str(course_meta.get("provas_ingresso_fonte", "")).strip()

        if fonte:
            lines.append(f"Fonte: {fonte}")

        return "\n".join(lines), [course_meta], [course_doc or ""]

    # 2. Se a ChromaDB ainda não tiver as provas, tenta ler diretamente do CSV
    csv_path = "data/estruturados/cursos_ipvc.csv"

    try:
        question_norm = normalize_text(question)
        question_words = get_meaningful_words(question)

        best_row = None
        best_score = 0

        with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                curso = str(row.get("curso", "")).strip()

                if not curso:
                    continue

                curso_norm = normalize_text(curso)
                curso_words = get_meaningful_words(curso)

                score = 0

                if curso_norm and curso_norm in question_norm:
                    score += 100

                overlap = question_words.intersection(curso_words)
                score += len(overlap) * 10

                if score > best_score:
                    best_score = score
                    best_row = row

        if best_row and best_score > 0:
            provas = str(best_row.get("provas_ingresso", "")).strip()
            fonte = str(best_row.get("provas_ingresso_fonte", "")).strip()
            course_name = str(best_row.get("curso", "")).strip()

            if provas:
                lines = [f"As provas de ingresso para {course_name} são:"]

                for prova in re.split(r";|\n|\|", provas):
                    prova = prova.strip(" -•\t")

                    if prova:
                        lines.append(f"- {prova}")

                if fonte:
                    lines.append(f"Fonte: {fonte}")

                csv_meta = dict(best_row)
                csv_meta["type"] = "structured_course"
                csv_meta["source"] = best_row.get("fonte", "cursos_ipvc.csv")

                csv_doc = (
                    f"Curso: {course_name}\n"
                    f"Provas de ingresso: {provas}\n"
                    f"Fonte das provas de ingresso: {fonte}"
                )

                return "\n".join(lines), [csv_meta], [csv_doc]

    except Exception:
        pass

    # 3. Última tentativa: procurar nos PDFs
    collection = get_collection()
    embedding_model = get_embedding_model()

    if course_name:
        query = f"{course_name} provas de ingresso exames nacionais acesso IPVC"
    else:
        query = f"{question} provas de ingresso exames nacionais acesso IPVC"

    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"type": "pdf_chunk"},
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return FALLBACK_ANSWER, [], []

    course_norm = normalize_text(course_name)

    candidate_indices = []

    if course_norm:
        for idx, doc in enumerate(documents):
            if course_norm in normalize_text(doc):
                candidate_indices.append(idx)

    if not candidate_indices:
        candidate_indices = list(range(len(documents)))

    exams = []
    used_exam_keys = set()

    for idx in candidate_indices:
        doc = documents[idx]
        doc_norm = normalize_text(doc)

        if course_norm and course_norm in doc_norm:
            pos = doc_norm.find(course_norm)
            snippet = doc[max(0, pos - 600):pos + 1200]
        else:
            snippet = doc

        for exam in extract_entry_exams_from_text(snippet):
            key = normalize_text(exam)

            if key not in used_exam_keys:
                used_exam_keys.add(key)
                exams.append(exam)

    if not exams:
        return FALLBACK_ANSWER, [], []

    if course_name:
        lines = [f"As provas de ingresso para {course_name} são:"]
    else:
        lines = ["Encontrei as seguintes provas de ingresso nos documentos disponíveis:"]

    for exam in exams[:6]:
        lines.append(f"- {exam}")

    returned_metas = metadatas
    returned_docs = documents

    if course_meta:
        returned_metas = [course_meta] + returned_metas
        returned_docs = [course_doc or ""] + returned_docs

    return "\n".join(lines), returned_metas, returned_docs


def is_transfer_question(question: str) -> bool:
    question_norm = normalize_text(question)

    keywords = [
        "mudanca de curso",
        "mudança de curso",
        "transferencia",
        "transferência",
        "mudar de curso",
        "trocar de curso",
        "reingresso",
        "concurso especial",
        "concursos especiais",
        "regime de mudança",
        "par instituição curso",
        "par instituicao curso",
        "mudança par instituição curso",
        "mudanca par instituicao curso",
    ]

    return any(normalize_text(keyword) in question_norm for keyword in keywords)


def get_pdf_chunks_by_keywords(keywords: List[str], max_chunks: int = 6) -> Tuple[List[str], List[dict]]:
    documents, metadatas = get_all_pdf_documents()
    scored = []

    normalized_keywords = [normalize_text(keyword) for keyword in keywords]

    for doc, meta in zip(documents, metadatas):
        source = str(meta.get("source", ""))
        searchable = normalize_text(source + "\n" + doc)
        score = 0

        for keyword in normalized_keywords:
            if keyword and keyword in searchable:
                score += 3

        if "regulamento" in searchable:
            score += 1
        if "reingresso" in searchable:
            score += 2
        if "mudanca" in searchable or "mudança" in searchable:
            score += 2
        if "transferencia" in searchable or "transferência" in searchable:
            score += 2

        if score > 0:
            scored.append((score, doc, meta))

    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    selected = scored[:max_chunks]

    return [doc for _, doc, _ in selected], [meta for _, _, meta in selected]


def extract_relevant_sentences_from_documents(
    documents: List[str],
    keywords: List[str],
    max_sentences: int = 5
) -> List[str]:
    normalized_keywords = [normalize_text(keyword) for keyword in keywords]
    selected = []
    seen = set()

    for doc in documents:
        parts = re.split(r"(?<=[.!?])\s+|\n+", doc)

        for part in parts:
            sentence = re.sub(r"\s+", " ", part).strip(" -•\t")

            if len(sentence) < 45 or len(sentence) > 450:
                continue

            sentence_norm = normalize_text(sentence)

            if any(keyword in sentence_norm for keyword in normalized_keywords):
                key = sentence_norm[:160]

                if key not in seen:
                    seen.add(key)
                    selected.append(sentence)

            if len(selected) >= max_sentences:
                return selected

    return selected


def answer_transfer_question(question: str) -> Tuple[str, List[dict], List[str]]:
    keywords = [
        "mudança de curso",
        "mudanca de curso",
        "transferência",
        "transferencia",
        "reingresso",
        "concurso especial",
        "concursos especiais",
        "mudança de par instituição curso",
        "mudanca de par instituicao curso",
        "par instituição curso",
        "par instituicao curso",
    ]

    documents, metadatas = get_pdf_chunks_by_keywords(keywords, max_chunks=8)

    if not documents:
        return FALLBACK_ANSWER, [], []

    sentences = extract_relevant_sentences_from_documents(
        documents=documents,
        keywords=keywords,
        max_sentences=5,
    )

    if not sentences:
        return FALLBACK_ANSWER, [], []

    lines = ["Com base nos documentos disponíveis, encontrei esta informação sobre mudança de curso/transferência:"]

    for sentence in sentences:
        lines.append(f"- {sentence}")

    return "\n".join(lines), metadatas, documents


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
            location=location,
        )

    collection = get_collection()
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    return results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]


def build_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)

    return f"""
És um assistente académico especializado nos cursos do IPVC.

Regras obrigatórias:
- Responde apenas com base no contexto fornecido.
- Não inventes informação.
- Nunca uses conhecimento externo.
- Se a resposta não estiver claramente no contexto, responde apenas: "{FALLBACK_ANSWER}"
- Não faças suposições.
- Não acrescentes exemplos inventados.
- Responde em português de Portugal.
- Sê claro e objetivo.
- Se a pergunta pedir listagens, usa tópicos.
- Se o contexto contiver uma lista completa, apresenta a lista completa.
- Não omitas elementos do contexto quando a pergunta pedir uma listagem completa.
- Se a pergunta estiver fora do âmbito do IPVC, responde exatamente: "{FALLBACK_ANSWER}"
- Se não existir informação suficiente no contexto, responde exatamente: "{FALLBACK_ANSWER}"
- Não expliques porque não encontraste informação.
- Não menciones instituições externas.
- Nunca alteres o significado da sigla IPVC. IPVC significa Instituto Politécnico de Viana do Castelo.

Contexto:
{context_text}

Pergunta:
{question}

Resposta:
""".strip()


def ask_bot(question: str) -> Tuple[str, List[dict], List[str]]:
    if is_out_of_scope_question(question):
        return FALLBACK_ANSWER, [], []

    # Respostas diretas e rápidas para escolas.
    if is_school_question(question):
        location = detect_location(question)

        if location:
            docs, metas = get_school_documents_by_location(location)
        else:
            docs, metas = get_all_school_documents()

        return build_direct_school_answer(metas, location=location), metas, docs

    if is_structured_courses_question(question):
        degree = detect_degree(question)
        school_code = detect_school_code(question)
        location = detect_location(question)
        docs, metas = get_courses_filtered(degree=degree, school_code=school_code, location=location)
        return build_direct_courses_answer(metas, degree=degree, school_code=school_code, location=location), metas, docs

    if is_entry_exam_question(question):
        return answer_entry_exam_question(question)

    if is_transfer_question(question):
        return answer_transfer_question(question)

    if is_comparison_question(question):
        return answer_comparison_question(question)

    if is_recommendation_question(question):
        context_chunks, metadatas = get_recommended_courses(question)
        return build_recommendation_answer(question, context_chunks, metadatas)

    if is_admission_question(question):
        return answer_admission_question(question)

    if is_career_question(question):
        context_chunks, metadatas = match_courses_for_career_question(question)
        answer = build_career_answer(question, metadatas)
        return answer, metadatas, context_chunks

    context_chunks, metadatas = search_relevant_context(question)

    if not context_chunks:
        return FALLBACK_ANSWER, [], []

    prompt = build_prompt(question, context_chunks)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_predict": 350,
            "num_ctx": 4096,
        },
    )

    answer = clean_portuguese_pt(response["message"]["content"])
    return answer, metadatas, context_chunks
