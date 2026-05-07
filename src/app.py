import os
import random
import html
import streamlit as st
from query_bot import ask_bot

st.set_page_config(
    page_title="IPVC Genius | Assistente Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def init_session_state():
    defaults = {
        "page": "home",
        "last_question": "",
        "last_answer": "",
        "last_metadatas": [],
        "last_context_chunks": [],
        "current_question": "",
        "auto_submit_question": False,
        "reset_current_question": False,
        "quick_suggestions": []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_page(page_name):
    st.session_state["page"] = page_name
    st.rerun()


def set_custom_style():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1e293b;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            background: radial-gradient(circle at top right, #f8fafc, #e2e8f0);
        }

        .main-card {
            background: white;
            padding: 3rem;
            border-radius: 32px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.7);
        }

        .hero-text-huge {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #0054A4, #00AA85);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
            margin-bottom: 1.5rem;
        }

        .stButton>button {
            border-radius: 16px !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            background: #0054A4 !important;
            color: white !important;
            font-weight: 600 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(0, 84, 164, 0.15) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(0, 84, 164, 0.25) !important;
            background: #004485 !important;
        }

        .stTextInput>div>div>input {
            border-radius: 16px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 1rem !important;
            transition: border-color 0.3s !important;
        }

        .stTextInput>div>div>input:focus {
            border-color: #0054A4 !important;
            box-shadow: none !important;
        }

        .bot-bubble {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 2rem;
            border-radius: 24px;
            border-bottom-left-radius: 4px;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
            margin-top: 1.5rem;
            position: relative;
        }

        .question-bubble {
            background: #f8fafc;
            border: 1px solid #dbe4ef;
            padding: 1.5rem;
            border-radius: 22px;
            box-shadow: 0 8px 18px -8px rgba(15, 23, 42, 0.12);
            margin-top: 2rem;
        }

        .bubble-title {
            font-weight: 800;
            color: #0054A4;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .suggestions-box {
            background: rgba(255,255,255,0.72);
            border: 1px solid #e2e8f0;
            border-radius: 24px;
            padding: 1.5rem;
            box-shadow: 0 15px 35px -18px rgba(15, 23, 42, 0.25);
            margin-top: -3.5rem;
        }

        .suggestions-title {
            font-weight: 800;
            color: #0054A4;
            font-size: 1.05rem;
            margin-bottom: 0.35rem;
        }

        .suggestions-subtitle {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .source-badge {
            background: #f1f5f9;
            color: #475569;
            padding: 4px 12px;
            border-radius: 99px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
            border: 1px solid #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)


def format_source_name(source: str) -> str:
    source = str(source).strip()

    if not source:
        return "Documento local"

    source_lower = source.lower()

    if "ctesp" in source_lower:
        return "Página oficial dos CTeSP do IPVC"

    if "mestrados" in source_lower:
        return "Página oficial dos Mestrados do IPVC"

    if "pos-graduacoes" in source_lower or "pos-gradua" in source_lower:
        return "Página oficial das Pós-graduações do IPVC"

    if "licenciaturas" in source_lower:
        return "Brochura de Licenciaturas do IPVC"

    if source_lower.endswith(".pdf"):
        return source

    if source.startswith("http"):
        return "Página oficial do IPVC"

    return source


def get_unique_sources(metadatas):
    sources = []

    for meta in metadatas:
        source = meta.get("source", "")

        if not source:
            continue

        label = format_source_name(source)
        url = source if str(source).startswith("http") else None

        item = (label, url)

        if item not in sources:
            sources.append(item)

    return sources


def render_sources(metadatas):
    sources = get_unique_sources(metadatas)

    if not sources:
        return

    with st.expander("📚 Ver fontes consultadas"):
        for label, url in sources:
            if url:
                st.markdown(f"- [{label}]({url})")
            else:
                st.markdown(f"- {label}")


def get_suggestion_pool():
    return [
        ("Ver cursos da ESCE", "Que cursos existem na ESCE?"),
        ("Ver cursos da ESTG", "Que cursos existem na ESTG?"),
        ("Ver cursos em Valença", "Que cursos existem em Valença?"),
        ("Ver cursos em Viana do Castelo", "Que cursos existem em Viana do Castelo?"),
        ("Ver mestrados da ESTG", "Que mestrados existem na ESTG?"),
        ("Ver mestrados do IPVC", "Que mestrados existem no IPVC?"),
        ("Ver CTeSP em Valença", "Que CTeSP existem em Valença?"),
        ("Ver CTeSP da ESTG", "Que CTeSP existem na ESTG?"),
        ("Ver pós-graduações da ESS", "Que pós-graduações existem na ESS?"),
        ("Ver licenciaturas em Viana", "Que licenciaturas existem em Viana do Castelo?"),
        ("Quero um curso de programação", "Gosto de programação, que curso recomendas?"),
        ("Quero um curso ligado a animais", "Gosto de animais, que cursos aconselhas?"),
        ("Quero um curso na área da saúde", "Quero trabalhar na área da saúde, que cursos aconselhas?"),
        ("Quero um curso de gestão", "Gosto de gestão e empresas, que curso devo escolher?"),
        ("Quero um curso de jogos digitais", "Tenho interesse em jogos digitais, que curso recomendas?"),
        ("Quero um curso ligado ao desporto", "Gosto de desporto, que cursos existem para mim?")
    ]


def refresh_quick_suggestions():
    pool = get_suggestion_pool()
    st.session_state["quick_suggestions"] = random.sample(pool, k=4)


def render_quick_questions_vertical():
    if not st.session_state["quick_suggestions"]:
        refresh_quick_suggestions()

    st.markdown("""
    <div class="suggestions-box">
        <div class="suggestions-title">Perguntas de exemplo</div>
        <div class="suggestions-subtitle">
            Estas sugestões referem-se a cursos, CTeSP, licenciaturas, mestrados e recomendações do IPVC.
            Clica numa opção para fazer a pergunta automaticamente.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height: 0.75rem;"></div>', unsafe_allow_html=True)

    suggestions = st.session_state["quick_suggestions"]

    for idx, (label, question) in enumerate(suggestions):
        if st.button(label, key=f"quick_{idx}_{label}", use_container_width=True):
            st.session_state["current_question"] = question
            st.session_state["auto_submit_question"] = True
            refresh_quick_suggestions()
            st.rerun()


def render_home():
    st.markdown(
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3rem;">'
        '<span style="font-weight: 800; color: #0054A4; font-size: 1.5rem;">IPVC.genius</span>'
        '<span style="color: #64748b;">CP2B AOOP 2026</span></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        st.markdown(
            '<div class="hero-text-huge">Dá o próximo passo na tua carreira académica.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="font-size: 1.2rem; color: #475569; margin-bottom: 2.5rem;">'
            'Consulta informações sobre cursos, pré-requisitos e escolas do Instituto Politécnico de Viana do Castelo '
            'através do nosso motor de inteligência artificial soberano e local.</p>',
            unsafe_allow_html=True
        )
        
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("🚀 Começar Conversa", use_container_width=True):
                refresh_quick_suggestions()
                go_to_page("chat")

        with btn_col2:
            if st.button("🛠️ Ver Arquitetura", use_container_width=True):
                go_to_page("about")

    with col2:
        hero_path = os.path.join("assets", "ipvc.webp")
        if os.path.exists(hero_path):
            st.markdown(
                f'<img src="data:image/webp;base64,{get_img_as_base64(hero_path)}" '
                f'style="width:100%; border-radius:30px; box-shadow: 20px 20px 60px #bebebe, -20px -20px 60px #ffffff;">',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#cbd5e1; height:400px; border-radius:30px; display:flex; '
                'align-items:center; justify-content:center; color:white;">Imagem: assets/ipvc.webp</div>',
                unsafe_allow_html=True
            )


def render_chat():
    if st.button("← Voltar"):
        st.session_state["quick_suggestions"] = []
        go_to_page("home")

    if st.session_state.get("reset_current_question", False):
        st.session_state["current_question"] = ""
        st.session_state["reset_current_question"] = False
        st.session_state["auto_submit_question"] = False

    chat_col, spacer_col, suggestions_col, right_space = st.columns(
        [2.0, 0.15, 0.85, 0.35],
        gap="large"
    )

    # Renderizar primeiro as sugestões para evitar erro ao alterar current_question
    # antes do st.text_input ser criado.
    with suggestions_col:
        render_quick_questions_vertical()

    with chat_col:
        st.markdown(
            '<h1 style="font-weight:800; margin-bottom:1.5rem;">Assistente Digital</h1>',
            unsafe_allow_html=True
        )

        question = st.text_input(
            "Pergunta",
            placeholder="Escreve aqui a tua pergunta sobre o IPVC...",
            label_visibility="collapsed",
            key="current_question"
        )

        st.markdown(
            '<p style="font-size:0.9rem; color:#64748b; margin-top:0.4rem; margin-bottom:1rem;">'
            'As respostas são geradas com base nos ficheiros locais carregados no sistema.'
            '</p>',
            unsafe_allow_html=True
        )
        
        col_send, col_clear, _ = st.columns([1, 1, 2])

        with col_send:
            ask_clicked = st.button("Enviar Pergunta", use_container_width=True)

        with col_clear:
            if st.button("Limpar Chat", use_container_width=True):
                st.session_state["last_question"] = ""
                st.session_state["last_answer"] = ""
                st.session_state["last_metadatas"] = []
                st.session_state["last_context_chunks"] = []
                st.session_state["auto_submit_question"] = False
                st.session_state["reset_current_question"] = True
                st.rerun()

        auto_submit = st.session_state.get("auto_submit_question", False)

        if ask_clicked and not question.strip():
            st.warning("Escreve uma pergunta antes de enviar.")

        if (ask_clicked or auto_submit) and question.strip():
            st.session_state["auto_submit_question"] = False

            with st.status("A consultar documentos ...", expanded=True) as status:
                try:
                    answer, metadatas, context_chunks = ask_bot(question)

                    st.session_state["last_question"] = question
                    st.session_state["last_answer"] = answer
                    st.session_state["last_metadatas"] = metadatas
                    st.session_state["last_context_chunks"] = context_chunks

                    status.update(label="Resposta gerada!", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"Erro na ligação ao modelo: {e}")

        if st.session_state["last_answer"]:
            safe_answer = html.escape(st.session_state["last_answer"]).replace("\n", "<br>")

            st.markdown(f"""
            <div class="bot-bubble">
                <div class="bubble-title">
                    <span>🎓</span> RESPOSTA
                </div>
                <div style="line-height:1.9; color:#334155;">
                    {safe_answer}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state["last_metadatas"]:
                render_sources(st.session_state["last_metadatas"])


def render_about():
    if st.button("← Voltar"):
        go_to_page("home")

    st.markdown(
        '<h1 style="font-weight:800;">Arquitetura do Sistema</h1>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p style="font-size:1.05rem; color:#475569; margin-bottom:2rem;">'
        'Esta aplicação utiliza uma abordagem RAG, combinando dados estruturados e não estruturados '
        'para responder a perguntas sobre cursos e formações do IPVC.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="background:white; border:1px solid #e2e8f0; border-radius:28px; '
        'padding:2rem; box-shadow:0 15px 35px -18px rgba(15,23,42,0.25); margin-bottom:2rem;">'
        '<h3 style="color:#0054A4; margin-top:0;">Fluxo RAG da aplicação</h3>'

        '<div style="font-size:1rem; line-height:2; color:#334155;">'

        '<strong>1. Dados locais</strong><br>'
        'CSVs com cursos, escolas, graus de ensino, localizações e áreas + PDFs/brochuras do IPVC'
        '<br><br>'

        '<strong>2. Pré-processamento</strong><br>'
        'Os dados são carregados, limpos e transformados em blocos de texto pesquisáveis'
        '<br><br>'

        '<strong>3. Embeddings</strong><br>'
        'Cada bloco é convertido num vetor usando SentenceTransformers'
        '<br><br>'

        '<strong>4. Base vetorial ChromaDB</strong><br>'
        'Os vetores e metadados são guardados localmente numa coleção chamada <code>ipvc_courses</code>'
        '<br><br>'

        '<strong>5. Pesquisa de contexto</strong><br>'
        'Quando o utilizador faz uma pergunta, o sistema procura os blocos mais relevantes'
        '<br><br>'

        '<strong>6. Modelo local Llama3 via Ollama</strong><br>'
        'O contexto recuperado é enviado para o modelo, que gera uma resposta em linguagem natural'
        '<br><br>'

        '<strong>7. Resposta final</strong><br>'
        'A resposta é apresentada ao utilizador juntamente com as fontes consultadas'

        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Dados estruturados**\n\n"
            "- cursos_ipvc.csv\n"
            "- escolas_ipvc.csv\n"
            "- Metadados: curso, escola, grau, local, regime, área, estado e fonte"
        )

    with col2:
        st.success(
            "**Dados não estruturados**\n\n"
            "- PDFs e brochuras do IPVC\n"
            "- Texto extraído e dividido em blocos\n"
            "- Pesquisa semântica através da ChromaDB"
        )

    st.markdown(
        '<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:22px; '
        'padding:1.5rem; margin-top:1.5rem;">'
        '<h4 style="color:#0054A4; margin-top:0;">Tecnologias utilizadas</h4>'
        '<p style="color:#334155; line-height:1.8; margin-bottom:0;">'
        'Python · Streamlit · ChromaDB · SentenceTransformers · Ollama · Llama3 · Pandas · PyPDF'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )


def get_img_as_base64(file):
    import base64

    with open(file, "rb") as f:
        data = f.read()

    return base64.b64encode(data).decode()


def main():
    init_session_state()
    set_custom_style()

    if st.session_state["page"] == "home":
        render_home()
    elif st.session_state["page"] == "chat":
        render_chat()
    elif st.session_state["page"] == "about":
        render_about()


if __name__ == "__main__":
    main()