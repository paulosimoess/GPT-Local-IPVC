import os
import streamlit as st

from query_bot import ask_bot


st.set_page_config(
    page_title="GPT Local IPVC",
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
        "last_context_chunks": []
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
        .stApp {
            background: linear-gradient(180deg, #f4f7fb 0%, #edf2f8 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
        }

        .brand {
            font-size: 1.1rem;
            font-weight: 700;
            color: #16324f;
        }

        .hero-wrap {
            background: linear-gradient(135deg, #0f4c81 0%, #2f74b5 100%);
            border-radius: 24px;
            padding: 2rem;
            color: white;
            box-shadow: 0 15px 35px rgba(19, 58, 102, 0.20);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.8rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            opacity: 0.95;
            margin-bottom: 1.5rem;
            max-width: 90%;
        }

        .card {
            background: white;
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 8px 24px rgba(20, 40, 80, 0.08);
            border: 1px solid rgba(28, 64, 102, 0.06);
            margin-bottom: 1rem;
        }

        .card h3 {
            margin-top: 0;
            margin-bottom: 0.4rem;
            color: #16324f;
        }

        .muted {
            color: #607086;
            font-size: 0.96rem;
        }

        .feature-card {
            background: white;
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 8px 22px rgba(20, 40, 80, 0.07);
            border: 1px solid #e6edf6;
            height: 100%;
        }

        .feature-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #16324f;
            margin-bottom: 0.35rem;
        }

        .answer-card {
            background: white;
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 8px 22px rgba(20, 40, 80, 0.08);
            border-left: 6px solid #0f4c81;
            margin-top: 1rem;
        }

        .source-box {
            background: #f4f7fb;
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.5rem;
            border: 1px solid #e3ebf5;
        }

        .step {
            background: white;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 18px rgba(20, 40, 80, 0.07);
            border-left: 5px solid #2f74b5;
            margin-bottom: 0.8rem;
        }

        .flow-box {
            background: white;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 6px 18px rgba(20, 40, 80, 0.06);
            text-align: center;
            font-weight: 600;
            color: #16324f;
            border: 1px solid #e6edf6;
        }

        .arrow {
            text-align: center;
            font-size: 1.8rem;
            color: #2f74b5;
            margin-top: 0.35rem;
        }

        div[data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)


def render_topbar():
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown('<div class="brand">🎓 GPT Local IPVC</div>', unsafe_allow_html=True)

    with col2:
        if st.button("Início", use_container_width=True):
            go_to_page("home")


def render_home():
    render_topbar()

    col_left, col_right = st.columns([1.15, 0.85], gap="large")

    with col_left:
        st.markdown("""
        <div class="hero-wrap">
            <div class="hero-title">Consulta e recomendação de cursos do IPVC com GPT Local</div>
            <div class="hero-subtitle">
                Plataforma académica com RAG, dados estruturados e documentos oficiais do IPVC,
                capaz de responder a perguntas e apoiar a descoberta de cursos de forma inteligente.
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💬 Abrir chatbot", use_container_width=True):
                go_to_page("chat")
        with c2:
            if st.button("🔎 Ver fluxo de resposta", use_container_width=True):
                go_to_page("about")

    with col_right:
        hero_path = os.path.join("assets", "ipvc.webp")

        if os.path.exists(hero_path):
            st.image(hero_path, use_container_width=True)
        else:
            st.markdown("""
            <div class="card" style="height:100%; min-height:320px; display:flex; align-items:center; justify-content:center; text-align:center;">
                <div>
                    <div style="font-size:4rem;">🎓</div>
                    <h3 style="margin-bottom:0.5rem;">Imagem principal do projeto</h3>
                    <p class="muted">
                        Adiciona uma imagem em <strong>assets/ipvc.webp</strong> para tornar a página inicial
                        mais próxima do estilo de landing page.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### O que podes fazer nesta plataforma")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Consultar cursos</div>
            <div class="muted">Perguntar por cursos, escolas, localizações, duração, áreas e saídas profissionais.</div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Receber recomendações</div>
            <div class="muted">Obter sugestões de cursos adequadas a interesses como programação, gestão, saúde ou educação.</div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Perceber a lógica do sistema</div>
            <div class="muted">Visualizar como os documentos são processados e como o sistema encontra a resposta final.</div>
        </div>
        """, unsafe_allow_html=True)


def render_chat():
    render_topbar()

    st.markdown("""
    <div class="card">
        <h2 style="margin-top:0;">Chatbot de cursos do IPVC</h2>
        <p class="muted">
            Coloca perguntas sobre cursos, escolas, áreas de formação, duração, localização ou saídas profissionais.
        </p>
    </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "Pergunta",
        value=st.session_state["last_question"],
        placeholder="Ex.: Que escolas do IPVC existem?"
    )

    c1, c2, c3 = st.columns([2.2, 1.2, 1.2])

    with c1:
        ask_clicked = st.button("Obter resposta", use_container_width=True)
    with c2:
        if st.button("Voltar", use_container_width=True):
            go_to_page("home")
    with c3:
        clear_clicked = st.button("Limpar", use_container_width=True)

    if clear_clicked:
        st.session_state["last_question"] = ""
        st.session_state["last_answer"] = ""
        st.session_state["last_metadatas"] = []
        st.session_state["last_context_chunks"] = []
        st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Escreve uma pergunta antes de continuar.")
        else:
            with st.spinner("A procurar contexto e a gerar resposta..."):
                try:
                    answer, metadatas, context_chunks = ask_bot(question)
                    st.session_state["last_question"] = question
                    st.session_state["last_answer"] = answer
                    st.session_state["last_metadatas"] = metadatas
                    st.session_state["last_context_chunks"] = context_chunks
                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")

    if st.session_state["last_answer"]:
        st.markdown("""
        <div class="answer-card">
            <h3 style="margin-top:0;">Resposta</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write(st.session_state["last_answer"])

        with st.expander("Fontes recuperadas"):
            metadatas = st.session_state["last_metadatas"]
            if not metadatas:
                st.info("Não foram apresentadas fontes.")
            else:
                for i, meta in enumerate(metadatas, start=1):
                    source = meta.get("source", "desconhecida")
                    doc_type = meta.get("type", "desconhecido")
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>{i}.</strong> {source}<br>
                            <span class="muted">{doc_type}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with st.expander("Contexto recuperado"):
            chunks = st.session_state["last_context_chunks"]
            if not chunks:
                st.info("Não foi recuperado contexto.")
            else:
                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**Bloco {i}**")
                    st.write(chunk[:700] + "..." if len(chunk) > 700 else chunk)


def render_about():
    render_topbar()

    st.markdown("""
    <div class="card">
        <h2 style="margin-top:0;">Como o sistema encontra as respostas</h2>
        <p class="muted">
            O sistema combina ficheiros CSV e documentos PDF, transforma essa informação em embeddings,
            pesquisa semanticamente o contexto mais relevante e só depois gera a resposta com um modelo local.
        </p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1. Recolha de dados", "São usados dados estruturados (cursos e escolas em CSV) e dados não estruturados (brochuras, guias e regulamentos em PDF)."),
        ("2. Preparação da informação", "O texto dos PDFs é extraído e dividido em blocos. Os dados do CSV são convertidos em descrições pesquisáveis."),
        ("3. Criação de embeddings", "Cada bloco é transformado numa representação vetorial, permitindo comparar semanticamente perguntas e conteúdos."),
        ("4. Pesquisa semântica", "Quando o utilizador escreve uma pergunta, o sistema procura no ChromaDB os blocos mais relevantes."),
        ("5. Geração da resposta", "O contexto recuperado é enviado ao modelo local no Ollama, que produz a resposta final."),
        ("6. Apresentação", "O utilizador recebe a resposta e pode ainda consultar fontes recuperadas e contexto usado.")
    ]

    for title, desc in steps:
        st.markdown(
            f"""
            <div class="step">
                <strong>{title}</strong><br>
                {desc}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Fluxo resumido")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown('<div class="flow-box">CSV + PDFs</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="flow-box">Preparação e chunks</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="flow-box">Embeddings + ChromaDB</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="flow-box">Pesquisa semântica</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="flow-box">Ollama + resposta</div>', unsafe_allow_html=True)


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