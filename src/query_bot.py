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
FALLBACK_ANSWER = "Não encontrei essa informação nos documentos disponíveis."


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)


# ============================================================
# Normalização e deteção simples
# ============================================================

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


# ============================================================
# Leitura de documentos estruturados da ChromaDB
# ============================================================

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
    location: Optional[str] = None,
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


# ============================================================
# Escolas e listagens diretas de cursos
# ============================================================

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


# ============================================================
# Médias, acesso e candidatura
# ============================================================

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
        return f"{number:.1f} valores (equivalente a {number / 10:.2f}/20)"

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


def get_meaningful_words(text: str) -> set:
    text_norm = normalize_text(text)

    stopwords = {
        "de", "da", "do", "das", "dos", "e", "em", "a", "o", "as", "os",
        "curso", "cursos", "licenciatura", "mestrado", "ctesp", "ipvc",
        "media", "nota", "ultimo", "colocado", "vagas", "quais", "qual",
        "para", "com", "uma", "um", "que", "como", "existem",
    }

    words = re.findall(r"\b[a-z0-9]+\b", text_norm)
    return {word for word in words if word not in stopwords and len(word) > 2}


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
        meta_school = meta.get("escola", "")
        meta_degree = meta.get("grau", "")
        meta_phase = meta.get("fase", "")

        if school_code and normalize_text(meta_school) != normalize_text(school_code):
            continue

        if degree and normalize_text(meta_degree) != normalize_text(degree):
            continue

        if phase and normalize_text(phase) != normalize_text(meta_phase):
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
        header_details.append(escola)
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


# ============================================================
# Recomendações por interesse e média
# ============================================================

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
        "seguranca": "ciberseguranca seguranca informatica redes sistemas",
    }

    for keyword, expansion in interest_map.items():
        if keyword in question_norm:
            expansions.append(expansion)

    if expansions:
        return question + " " + " ".join(expansions)

    return question


def get_recommended_courses(question: str, n_results: int = 8) -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    embedding_model = get_embedding_model()

    expanded_question = expand_recommendation_query(question)
    query_embedding = embedding_model.encode([expanded_question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"type": "structured_course"},
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return documents, metadatas


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
        phase = normalize_text(meta.get("fase", ""))
        phase_priority = 0 if "1." in phase or "1ª" in phase or "1a" in phase else 1
        grade = parse_number(meta.get("nota_ultimo_colocado_contingente_geral", ""))
        grade_200 = normalize_grade_to_200(grade) if grade is not None else 9999
        return phase_priority, grade_200

    return sorted(candidates, key=sort_key)[0]


def build_recommendation_answer(question: str, course_docs: List[str], course_metas: List[dict]) -> Tuple[str, List[dict], List[str]]:
    if not course_metas:
        return FALLBACK_ANSWER, [], []

    threshold = detect_average_threshold(question)
    preferred_phase = detect_phase(question)

    enriched = []
    used_keys = set()

    for doc, meta in zip(course_docs, course_metas):
        curso = meta.get("curso", "")
        grau = meta.get("grau", "")
        escola = meta.get("escola", "")

        if not curso:
            continue

        key = (normalize_text(curso), normalize_text(grau), normalize_text(escola))
        if key in used_keys:
            continue
        used_keys.add(key)

        admission = find_best_admission_for_course(meta, preferred_phase=preferred_phase)
        grade = None

        if admission:
            grade = parse_number(admission.get("nota_ultimo_colocado_contingente_geral", ""))
            if grade is not None:
                grade = normalize_grade_to_200(grade)

        if threshold is None:
            score_group = 0
        elif grade is None:
            score_group = 1
        elif grade <= threshold:
            score_group = 0
        else:
            score_group = 2

        distance = abs((grade or threshold or 0) - (threshold or grade or 0)) if grade is not None and threshold is not None else 0

        enriched.append({
            "doc": doc,
            "course_meta": meta,
            "admission_meta": admission,
            "grade": grade,
            "score_group": score_group,
            "distance": distance,
        })

    enriched = sorted(enriched, key=lambda item: (item["score_group"], item["distance"]))
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
        grau = meta.get("grau", "")
        escola = meta.get("escola", "")
        local = meta.get("local", "")
        resumo = meta.get("resumo", "") or meta.get("descricao", "")
        area = meta.get("area", "")

        details = []
        if grau:
            details.append(grau)
        if escola:
            details.append(escola)
        if local:
            details.append(local)

        if resumo:
            explanation = resumo
        elif area:
            explanation = f"é uma opção relacionada com a área de {area}"
        else:
            explanation = "é uma opção relacionada com o interesse indicado"

        line = f"- {curso}"

        if details:
            line += f" ({', '.join(details)})"

        line += f" — {explanation}"

        if admission and grade is not None:
            line += f" Nota do último colocado: {format_grade(admission.get('nota_ultimo_colocado_contingente_geral', ''))}."

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


# ============================================================
# Saídas profissionais
# ============================================================

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
        area = meta.get("area", "")
        saidas = meta.get("saidas_profissionais", "")
        interesses = meta.get("interesses_relacionados", "")
        palavras = meta.get("palavras_chave", "")

        searchable_text = " ".join([curso, area, saidas, interesses, palavras])
        searchable_norm = normalize_text(searchable_text)
        course_norm = normalize_text(curso)

        if not curso:
            continue

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

    documents = [doc for _, doc, _ in selected]
    metadatas = [meta for _, _, meta in selected]

    return documents, metadatas


def build_career_answer(question: str, metadatas: List[dict]) -> str:
    if not metadatas:
        return FALLBACK_ANSWER

    question_norm = normalize_text(question)
    lines = []

    specific_course = None

    for meta in metadatas:
        curso = meta.get("curso", "")
        if curso and normalize_text(curso) in question_norm:
            specific_course = meta
            break

    if specific_course:
        curso = specific_course.get("curso", "")
        escola = specific_course.get("escola", "")
        grau = specific_course.get("grau", "")
        saidas = specific_course.get("saidas_profissionais", "")

        if not saidas:
            return FALLBACK_ANSWER

        lines.append(f"As saídas profissionais de {curso} são:")

        details = []
        if grau:
            details.append(grau)
        if escola:
            details.append(escola)

        if details:
            lines.append(f"({', '.join(details)})")

        for item in str(saidas).split(";"):
            item = item.strip()
            if item:
                lines.append(f"- {item}")

        return "\n".join(lines)

    lines.append("Encontrei os seguintes cursos relacionados com essa área profissional:")

    used = set()

    for meta in metadatas[:5]:
        curso = meta.get("curso", "")
        escola = meta.get("escola", "")
        grau = meta.get("grau", "")
        saidas = meta.get("saidas_profissionais", "")
        area = meta.get("area", "")

        if not curso or curso in used:
            continue

        used.add(curso)
        details = []

        if grau:
            details.append(grau)
        if escola:
            details.append(escola)

        info = curso

        if details:
            info += f" ({', '.join(details)})"

        if saidas:
            info += f" — saídas profissionais: {saidas}"
        elif area:
            info += f" — relacionado com a área de {area}"

        lines.append(f"- {info}")

    if len(lines) == 1:
        return FALLBACK_ANSWER

    return "\n".join(lines)


# ============================================================
# Comparação entre cursos
# ============================================================

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


def find_courses_for_comparison(question: str, max_courses: int = 3) -> Tuple[List[str], List[dict]]:
    all_documents, all_metadatas = get_all_course_documents()
    question_norm = normalize_text(question)

    exact_matches = []
    used_keys = set()

    for doc, meta in zip(all_documents, all_metadatas):
        curso = meta.get("curso", "")

        if not curso:
            continue

        curso_norm = normalize_text(curso)

        if curso_norm and curso_norm in question_norm:
            key = (
                normalize_text(meta.get("curso", "")),
                normalize_text(meta.get("grau", "")),
                normalize_text(meta.get("escola", "")),
            )

            if key not in used_keys:
                used_keys.add(key)
                exact_matches.append((len(curso_norm), doc, meta))

    exact_matches = sorted(exact_matches, key=lambda item: item[0], reverse=True)

    if len(exact_matches) >= 2:
        selected = exact_matches[:max_courses]
        documents = [doc for _, doc, _ in selected]
        metadatas = [meta for _, _, meta in selected]
        return documents, metadatas

    question_words = get_meaningful_words(question)
    scored = []

    for doc, meta in zip(all_documents, all_metadatas):
        curso = meta.get("curso", "")

        if not curso:
            continue

        key = (
            normalize_text(meta.get("curso", "")),
            normalize_text(meta.get("grau", "")),
            normalize_text(meta.get("escola", "")),
        )

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

    for _, _, doc, meta in scored:
        selected_docs.append(doc)
        selected_metas.append(meta)

        if len(selected_metas) >= max_courses:
            break

    if exact_matches:
        selected_docs = [doc for _, doc, _ in exact_matches[:1]] + selected_docs
        selected_metas = [meta for _, _, meta in exact_matches[:1]] + selected_metas

    return selected_docs[:max_courses], selected_metas[:max_courses]


def build_comparison_context(context_chunks: List[str], metadatas: List[dict]) -> str:
    lines = []

    for idx, (doc, meta) in enumerate(zip(context_chunks, metadatas), start=1):
        curso = meta.get("curso", "")
        grau = meta.get("grau", "")
        escola = meta.get("escola", "")
        area = meta.get("area", "")
        source = meta.get("source", "")

        lines.append(
            f"Curso {idx}:\n"
            f"Nome: {curso}\n"
            f"Grau: {grau}\n"
            f"Escola: {escola}\n"
            f"Área: {area}\n"
            f"Fonte: {source}\n"
            f"Informação disponível:\n{doc[:1800]}"
        )

    return "\n\n---\n\n".join(lines)


def build_comparison_prompt(question: str, comparison_context: str) -> str:
    return f"""
És um assistente académico especializado nos cursos e formações do IPVC.

Tarefa:
Comparar cursos com base apenas no contexto fornecido.

Regras obrigatórias:
- Usa apenas os cursos presentes no contexto.
- Não inventes cursos, dados, médias, saídas profissionais ou características.
- Não uses conhecimento externo.
- Se não houver pelo menos dois cursos no contexto, responde exatamente: "{FALLBACK_ANSWER}"
- Responde em português de Portugal.
- Nunca uses expressões brasileiras como "se concentra", "em uma" ou "em um".
- Usa frases impessoais sempre que possível.
- Não uses markdown com negrito.
- Não uses asteriscos.
- Sê claro, objetivo e útil para um candidato.
- Usa tópicos simples começados por "-".
- Não faças uma resposta demasiado longa.
- Não substituas um curso pedido por outro curso parecido sem dizer explicitamente que o curso pedido não foi encontrado.
- Na indicação final, usa sempre "Esta opção pode fazer mais sentido se..." em vez de frases com "tu" ou "você".

Estrutura obrigatória:
Começa exatamente com:
"Comparação entre os cursos:"

Depois usa este formato:

- Curso 1: nome do curso
Grau:
Escola:
Síntese:

- Curso 2: nome do curso
Grau:
Escola:
Síntese:

Principais diferenças:
- diferença 1
- diferença 2
- diferença 3

Indicação final:
- Esta opção pode fazer mais sentido se...
- A outra opção pode fazer mais sentido se...

Contexto:
{comparison_context}

Pergunta do utilizador:
{question}

Resposta:
""".strip()


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
        "através de uma": "através de uma",
    }

    cleaned = text

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
        cleaned = cleaned.replace(old.capitalize(), new.capitalize())

    return cleaned.strip()


def answer_comparison_question(question: str) -> Tuple[str, List[dict], List[str]]:
    context_chunks, metadatas = find_courses_for_comparison(question)

    if len(metadatas) < 2:
        return FALLBACK_ANSWER, [], []

    comparison_context = build_comparison_context(context_chunks, metadatas)
    prompt = build_comparison_prompt(question, comparison_context)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_predict": 650,
            "num_ctx": 4096,
        },
    )

    answer = clean_portuguese_pt(response["message"]["content"])
    return answer, metadatas, context_chunks


# ============================================================
# Provas de ingresso
# ============================================================

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


def answer_entry_exam_question(question: str) -> Tuple[str, List[dict], List[str]]:
    collection = get_collection()
    embedding_model = get_embedding_model()

    expanded_question = (
        question
        + " provas de ingresso exames nacionais candidatura acesso ensino superior IPVC curso"
    )

    query_embedding = embedding_model.encode([expanded_question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return FALLBACK_ANSWER, [], []

    context_text = "\n\n---\n\n".join(documents)

    prompt = f"""
És um assistente académico especializado no acesso aos cursos do IPVC.

Tarefa:
Responder a perguntas sobre provas de ingresso e exames nacionais.

Regras obrigatórias:
- Usa apenas o contexto fornecido.
- Não inventes provas de ingresso.
- Não uses conhecimento externo.
- Se o contexto não indicar claramente as provas de ingresso, responde exatamente: "{FALLBACK_ANSWER}"
- Responde em português de Portugal.
- Sê direto.
- Se encontrares a informação, indica o curso e as respetivas provas de ingresso.
- Usa tópicos simples.

Contexto:
{context_text}

Pergunta:
{question}

Resposta:
""".strip()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_predict": 400,
            "num_ctx": 4096,
        },
    )

    answer = clean_portuguese_pt(response["message"]["content"])
    return answer, metadatas, documents


# ============================================================
# Mudança de curso / transferência / reingresso
# ============================================================

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


def answer_transfer_question(question: str) -> Tuple[str, List[dict], List[str]]:
    collection = get_collection()
    embedding_model = get_embedding_model()

    expanded_question = (
        question
        + " mudança de curso transferência reingresso concurso especial regulamento candidatura IPVC par instituição curso"
    )

    query_embedding = embedding_model.encode([expanded_question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"type": "pdf_chunk"},
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return FALLBACK_ANSWER, [], []

    context_text = "\n\n---\n\n".join(documents)

    prompt = f"""
És um assistente académico especializado no IPVC.

Tarefa:
Responder a perguntas sobre mudança de curso, transferência, reingresso ou concursos especiais.

Regras obrigatórias:
- Usa apenas o contexto fornecido.
- Não inventes regras, prazos, documentos ou condições.
- Não uses conhecimento externo.
- Se o contexto não indicar claramente a resposta, responde exatamente: "{FALLBACK_ANSWER}"
- Responde em português de Portugal.
- Explica de forma simples e objetiva.
- Quando aplicável, refere que o candidato deve consultar os serviços académicos ou a página oficial do IPVC.
- Usa tópicos simples.

Contexto:
{context_text}

Pergunta:
{question}

Resposta:
""".strip()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_predict": 450,
            "num_ctx": 4096,
        },
    )

    answer = clean_portuguese_pt(response["message"]["content"])
    return answer, metadatas, documents


# ============================================================
# Pesquisa genérica RAG
# ============================================================

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


# Mantive estas duas funções para compatibilidade, caso as queiras usar mais tarde.
def build_compact_recommendation_context(metadatas: List[dict]) -> str:
    lines = []

    for idx, meta in enumerate(metadatas[:4], start=1):
        curso = meta.get("curso", "")
        grau = meta.get("grau", "")
        escola = meta.get("escola", "")
        area = meta.get("area", "")
        local = meta.get("local", "")
        resumo = meta.get("resumo", "")
        interesses = meta.get("interesses_relacionados", "")
        palavras_chave = meta.get("palavras_chave", "")
        saidas = meta.get("saidas_profissionais", "")

        if not curso:
            continue

        lines.append(
            f"Opção {idx}:\n"
            f"Curso: {curso}\n"
            f"Grau: {grau}\n"
            f"Escola: {escola}\n"
            f"Local: {local}\n"
            f"Área: {area}\n"
            f"Resumo: {resumo}\n"
            f"Interesses relacionados: {interesses}\n"
            f"Palavras-chave: {palavras_chave}\n"
            f"Saídas profissionais: {saidas}"
        )

    return "\n\n---\n\n".join(lines)


def build_recommendation_prompt(question: str, compact_context: str) -> str:
    return f"""
És um assistente académico especializado na recomendação de cursos e formações do IPVC.

Tarefa:
Recomendar cursos com base no interesse indicado pelo utilizador.

Regras obrigatórias:
- Começa sempre a resposta com a frase: "Com base no teu interesse, recomendo:"
- Usa apenas as opções presentes no contexto.
- Não inventes cursos.
- Não uses conhecimento externo.
- Recomenda no máximo 3 opções.
- Não apresentes cursos rejeitados.
- Para cada opção, escreve apenas uma explicação curta.
- Indica sempre o nome do curso, o grau e a escola.
- Responde em português de Portugal.
- Não uses "você".
- Usa linguagem natural em português europeu.
- Não uses expressões como "Espero que isso ajude".
- Não uses markdown com negrito.
- Usa apenas tópicos simples começados por "-".
- A resposta deve ser curta e direta.
- Usa apenas este formato: "- Nome do curso (Grau, Escola, Local) — explicação curta."
- Não escrevas campos separados como "Curso:", "Grau:", "Escola:", "Local:", "Área:" ou "Resumo:".
- Não repitas o mesmo curso com o mesmo grau.
- Não incluas introduções longas.

Contexto:
{compact_context}

Pergunta do utilizador:
{question}

Resposta:
""".strip()


# ============================================================
# Função principal chamada pela app
# ============================================================

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

    # Respostas diretas e completas para listagens de cursos.
    if is_structured_courses_question(question):
        degree = detect_degree(question)
        school_code = detect_school_code(question)
        location = detect_location(question)
        docs, metas = get_courses_filtered(degree=degree, school_code=school_code, location=location)
        return build_direct_courses_answer(metas, degree=degree, school_code=school_code, location=location), metas, docs

    # Provas de ingresso devem ser tratadas antes das médias gerais.
    if is_entry_exam_question(question):
        return answer_entry_exam_question(question)

    # Mudança de curso / transferência / reingresso.
    if is_transfer_question(question):
        return answer_transfer_question(question)

    # Recomendação vem antes da admissão, para perguntas como:
    # "Tenho média 14 e gosto de programação, que curso recomendas?"
    if is_recommendation_question(question):
        context_chunks, metadatas = get_recommended_courses(question)
        return build_recommendation_answer(question, context_chunks, metadatas)

    # Médias, vagas, notas do último colocado e dados DGES.
    if is_admission_question(question):
        return answer_admission_question(question)

    # Saídas profissionais.
    if is_career_question(question):
        context_chunks, metadatas = match_courses_for_career_question(question)
        answer = build_career_answer(question, metadatas)
        return answer, metadatas, context_chunks

    # Comparação entre cursos.
    if is_comparison_question(question):
        return answer_comparison_question(question)

    # RAG genérico para restantes perguntas sobre documentos IPVC.
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
