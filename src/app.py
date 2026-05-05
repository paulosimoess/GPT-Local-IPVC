import streamlit as st

from query_bot import ask_bot


st.set_page_config(
    page_title="GPT Local IPVC",
    page_icon="🎓",
    layout="wide"
)


def set_custom_style():
    st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .hero-box {
            background: linear-gradient(135deg, #0f4c81, #2d6ca2);
            padding: 2.2rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.95;
        }

        .section-card {
            background: white;
            padding: 1.4rem;
            border-radius: 16px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }

        .small-note {
            color: #5f6b7a;
            font-size: 0.95rem;
        }

        .step-box {
            background: #ffffff;
            border-left: 6px solid #0f4c81;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            margin-bottom: 0.8rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }

        .source-box {
            background: #f1f5f9;
            border-radius: 10px;
            padding: 0.8rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)


def go_to_page(page_name):
    st.session_state["page"] = page_name


def render_home():
    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">GPT Local para recomendação e consulta de cursos do IPVC</div>
        <div class="hero-subtitle">
            Projeto académico com recurso a um modelo local, pesquisa semântica e dados estruturados e não estruturados.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>Fazer pergunta ao bot</h3>
            <p class="small-note">
                Acede à área de interação com o assistente para colocar perguntas sobre cursos,
                escolas, duração, áreas de interesse e outras informações disponíveis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Ir para a área de perguntas", use_container_width=True):
            go_to_page("chat")

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>Como o sistema encontra as respostas</h3>
            <p class="small-note">
                Consulta um esquema explicativo do funcionamento do sistema, incluindo a utilização
                de PDFs, ficheiros CSV, embeddings, base vetorial e modelo local.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Ver explicação do sistema", use_container_width=True):
            go_to_page("about")

    st.markdown("""
    <div class="section-card">
        <h3>Objetivo do projeto</h3>
        <p class="small-note">
            Este sistema foi desenvolvido para responder a perguntas sobre cursos do IPVC e apoiar
            recomendações com base em dados estruturados e não estruturados, recorrendo a um GPT local
            e a uma abordagem RAG.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_chat():
    st.title("Área de perguntas ao bot")
    st.write("Coloca uma pergunta sobre cursos, escolas, áreas ou características da oferta formativa do IPVC.")

    question = st.text_input("Pergunta", placeholder="Ex.: Que cursos de informática existem no IPVC?")

    if st.button("Obter resposta", use_container_width=True):
        if not question.strip():
            st.warning("Escreve uma pergunta antes de continuar.")
        else:
            with st.spinner("A procurar contexto e a gerar resposta..."):
                try:
                    answer, metadatas, context_chunks = ask_bot(question)

                    st.markdown("""
                    <div class="section-card">
                        <h3>Resposta</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(answer)

                    with st.expander("Fontes recuperadas"):
                        if not metadatas:
                            st.info("Não foram apresentadas fontes.")
                        else:
                            for i, meta in enumerate(metadatas, start=1):
                                source = meta.get("source", "desconhecida")
                                doc_type = meta.get("type", "desconhecido")
                                st.markdown(
                                    "<div class='source-box'><strong>{0}.</strong> {1} <br><span class='small-note'>{2}</span></div>".format(
                                        i, source, doc_type
                                    ),
                                    unsafe_allow_html=True
                                )

                    with st.expander("Contexto recuperado"):
                        if not context_chunks:
                            st.info("Não foi recuperado contexto.")
                        else:
                            for i, chunk in enumerate(context_chunks, start=1):
                                st.markdown("**Bloco {0}**".format(i))
                                st.write(chunk)

                except Exception as e:
                    st.error("Ocorreu um erro: {0}".format(e))

    if st.button("Voltar à página inicial"):
        go_to_page("home")


def render_about():
    st.title("Como o sistema encontra as respostas")

    st.markdown("""
    <div class="step-box">
        <strong>1. Recolha de dados</strong><br>
        O sistema utiliza dois tipos de fontes: dados estruturados (ficheiros CSV com cursos e escolas)
        e dados não estruturados (documentos PDF do IPVC, como brochuras, guias e regulamentos).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-box">
        <strong>2. Extração e preparação</strong><br>
        O texto dos PDFs é extraído automaticamente e dividido em blocos menores. Os dados do CSV são
        transformados em texto descritivo para também poderem ser pesquisados.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-box">
        <strong>3. Criação de embeddings</strong><br>
        Cada bloco de texto é convertido numa representação vetorial (embedding), permitindo comparar
        semanticamente a pergunta do utilizador com os conteúdos armazenados.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-box">
        <strong>4. Pesquisa semântica</strong><br>
        Quando o utilizador faz uma pergunta, o sistema procura os blocos mais relevantes na base vetorial
        (ChromaDB), recuperando o contexto mais útil para responder.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-box">
        <strong>5. Geração da resposta</strong><br>
        O contexto recuperado é enviado para um modelo local no Ollama, que gera a resposta final com base
        apenas na informação disponível.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-box">
        <strong>6. Apresentação ao utilizador</strong><br>
        O utilizador vê a resposta, podendo também consultar as fontes recuperadas e os blocos de contexto
        utilizados pelo sistema.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Esquema resumido")
    st.code(
        "CSV + PDFs -> extração/preparação -> embeddings -> ChromaDB -> pergunta do utilizador -> "
        "pesquisa semântica -> Ollama -> resposta final",
        language="text"
    )

    if st.button("Voltar à página inicial"):
        go_to_page("home")


def main():
    set_custom_style()

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if st.session_state["page"] == "home":
        render_home()
    elif st.session_state["page"] == "chat":
        render_chat()
    elif st.session_state["page"] == "about":
        render_about()


if __name__ == "__main__":
    main()