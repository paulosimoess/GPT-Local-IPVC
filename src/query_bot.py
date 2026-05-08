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


def detect_phase(question: str) -> Optional[str]:
    question_norm = normalize_text(question)

    first_phase_patterns = [
        r"\b1\s*\.?\s*[ªa]?\s*fase\b",
        r"\bprimeira fase\b",
        r"\b1 fase\b"
    ]

    second_phase_patterns = [
        r"\b2\s*\.?\s*[ªa]?\s*fase\b",
        r"\bsegunda fase\b",
        r"\b2 fase\b"
    ]

    for pattern in first_phase_patterns:
        if re.search(pattern, question_norm):
            return "1.ª fase CNA"

    for pattern in second_phase_patterns:
        if re.search(pattern, question_norm):
            return "2.ª fase CNA"

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


def get_recommended_courses(question: str, n_results: int = 4) -> Tuple[List[str], List[dict]]:
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

def is_out_of_scope_question(question: str) -> bool:
    question_norm = normalize_text(question)

    # Perguntas sobre outras instituições ou temas claramente externos ao IPVC
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
        "porto",
        "sporting",
        "cinema",
        "filme",
        "musica",
        "receita",
        "cozinhar"
    ]

    for term in external_terms:
        if normalize_text(term) in question_norm:
            return True

    return False


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
        "provas de ingresso"
    ]

    return any(keyword in question_norm for keyword in keywords)


def get_all_admission_documents() -> Tuple[List[str], List[dict]]:
    collection = get_collection()
    results = collection.get(where={"type": "structured_admission"})
    return results.get("documents", []), results.get("metadatas", [])


def parse_number(value) -> Optional[float]:
    if value is None:
        return None

    value_str = str(value).strip().replace(",", ".")

    if not value_str:
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
        r"(?:media|nota)\s+(?:abaixo|inferior|menor|ate|até)\s+(?:de|a)?\s*(\d+(?:[.,]\d+)?)"
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
        "media", "nota", "ultimo", "colocado", "vagas"
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

    line += f"\n  • Nota do último colocado: {format_grade(nota)}"

    if vagas != "":
        line += f"\n  • Vagas iniciais: {vagas}"

    if colocados != "":
        line += f"\n  • Colocados: {colocados}"

    if sobras != "":
        line += f"\n  • Sobras para a 2.ª fase: {sobras}"

    return line


def answer_admission_question(question: str) -> Tuple[str, List[dict], List[str]]:
    question_norm = normalize_text(question)
    phase = detect_phase(question)

    if "provas de ingresso" in question_norm or "provas ingresso" in question_norm:
        return FALLBACK_ANSWER, [], []

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
- Se a pergunta estiver fora do âmbito do IPVC, responde exatamente: "Não encontrei essa informação nos documentos disponíveis."
- Se não existir informação suficiente no contexto, responde exatamente: "Não encontrei essa informação nos documentos disponíveis."
- Não expliques porque não encontraste informação.
- Não menciones instituições externas.
- Nunca alteres o significado da sigla IPVC. IPVC significa Instituto Politécnico de Viana do Castelo.

Contexto:
{context_text}

Pergunta:
{question}

Resposta:
""".strip()


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


def ask_bot(question: str) -> Tuple[str, List[dict], List[str]]:
    if is_out_of_scope_question(question):
        return FALLBACK_ANSWER, [], []

    if is_admission_question(question):
        return answer_admission_question(question)
    
    if is_recommendation_question(question):
        context_chunks, metadatas = get_recommended_courses(question)

        compact_context = build_compact_recommendation_context(metadatas)
        prompt = build_recommendation_prompt(question, compact_context)

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
        "temperature": 0.1,
        "num_predict": 200,
        "num_ctx": 1024
        }
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