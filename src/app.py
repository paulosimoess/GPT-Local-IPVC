import os
import random
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
            margin-top: 2rem;
            position: relative;
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
        ("Cursos ESCE", "Que cursos existem na ESCE?"),
        ("Cursos ESTG", "Que cursos existem na ESTG?"),
        ("Cursos Valença", "Que cursos existem em Valença?"),
        ("Cursos Viana", "Que cursos existem em Viana do Castelo?"),
        ("Mestrados ESTG", "Que mestrados existem na ESTG?"),
        ("Mestrados IPVC", "Que mestrados existem no IPVC?"),
        ("CTeSP Valença", "Que CTeSP existem em Valença?"),
        ("CTeSP ESTG", "Que CTeSP existem na ESTG?"),
        ("Pós-grad. ESS", "Que pós-graduações existem na ESS?"),
        ("Lic. Viana", "Que licenciaturas existem em Viana do Castelo?"),
        ("Programação", "Gosto de programação, que curso recomendas?"),
        ("Animais", "Gosto de animais, que cursos aconselhas?"),
        ("Saúde", "Quero trabalhar na área da saúde, que cursos aconselhas?"),
        ("Gestão", "Gosto de gestão e empresas, que curso devo escolher?"),
        ("Jogos Digitais", "Tenho interesse em jogos digitais, que curso recomendas?"),
        ("Desporto", "Gosto de desporto, que cursos existem para mim?")
    ]


def refresh_quick_suggestions():
    pool = get_suggestion_pool()
    st.session_state["quick_suggestions"] = random.sample(pool, k=4)


def render_quick_questions():
    if not st.session_state["quick_suggestions"]:
        refresh_quick_suggestions()

    st.markdown(
        '<p style="font-weight:700; color:#475569; margin-top:1rem; margin-bottom:0.75rem;">Sugestões rápidas:</p>',
        unsafe_allow_html=True
    )

    suggestions = st.session_state["quick_suggestions"]

    cols = st.columns(4, gap="medium")

    for idx, (label, question) in enumerate(suggestions):
        with cols[idx]:
            if st.button(label, key=f"quick_{idx}_{label}", use_container_width=True):
                st.session_state["current_question"] = question
                st.session_state["auto_submit_question"] = True
                refresh_quick_suggestions()
                st.rerun()

    st.markdown('<div style="height: 2.75rem;"></div>', unsafe_allow_html=True)


def render_home():
    st.markdown(
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3rem;">'
        '<span style="font-weight: 800; color: #0054A4; font-size: 1.5rem;">IPVC.genius</span>'
        '<span style="color: #64748b;">Trabalho Académico 2026</span></div>',
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
    st.markdown(
        '<p style="color:#0054A4; font-weight:700; cursor:pointer;" onclick="window.location.reload()">← VOLTAR AO INÍCIO</p>',
        unsafe_allow_html=True
    )

    if st.button("Voltar"):
        st.session_state["quick_suggestions"] = []
        go_to_page("home")

    if st.session_state.get("reset_current_question", False):
        st.session_state["current_question"] = ""
        st.session_state["reset_current_question"] = False
        st.session_state["auto_submit_question"] = False

    st.markdown('<h1 style="font-weight:800;">Assistente Digital</h1>', unsafe_allow_html=True)

    render_quick_questions()
    
    question = st.text_input(
        "Pergunta",
        placeholder="Escreve aqui a tua pergunta sobre o IPVC...",
        label_visibility="collapsed",
        key="current_question"
    )
    
    col_send, col_clear, _ = st.columns([1, 1, 3])

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

    if (ask_clicked or auto_submit) and question:
        st.session_state["auto_submit_question"] = False

        with st.status("🔍 A consultar documentos ...", expanded=True) as status:
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
        answer_html = st.session_state["last_answer"].replace("\n", "<br>")

        st.markdown(f"""
        <div class="bot-bubble">
            <div style="font-weight:800; color:#0054A4; margin-bottom:1rem; display:flex; align-items:center;">
                <span style="margin-right:10px;">🎓</span> RESPOSTA
            </div>
            <div style="line-height:1.9; color:#334155;">
                {answer_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state["last_metadatas"]:
            render_sources(st.session_state["last_metadatas"])


def render_about():
    if st.button("← Voltar"):
        go_to_page("home")

    st.markdown('<h1 style="font-weight:800;">Como funciona?</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.info("**RAG (Retrieval-Augmented Generation)**: O sistema não inventa. Ele lê os teus PDFs e CSVs primeiro.")

    with col2:
        st.success("**Privacidade Local**: O Llama3 corre no teu PC via Ollama. Nenhum dado sai do IPVC.")
    
    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/1*v6S_S_vsh_x-GfJ_v7Vf6w.png",
        caption="Esquema do fluxo RAG"
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