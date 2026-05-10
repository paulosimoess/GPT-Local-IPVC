# IPVC Genius - GPT Local para Consulta e Recomendação de Cursos

## Descrição do projeto

O **IPVC Genius** é uma aplicação académica desenvolvida no âmbito do trabalho prático CP2Bde AOOP, com o objetivo de criar uma plataforma local baseada em GPTs com RAG (*Retrieval-Augmented Generation*), aplicada ao contexto específico dos cursos e formações do Instituto Politécnico de Viana do Castelo.

A aplicação permite ao utilizador consultar informação sobre cursos, escolas, localizações, graus de ensino, médias de acesso, provas de ingresso, saídas profissionais e recomendações com base em interesses pessoais. Para isso, utiliza dados estruturados e não estruturados carregados localmente.

## Tema escolhido

**GPT Local para recomendação e consulta de cursos do IPVC**

## Objetivo

O objetivo principal é criar um assistente digital capaz de responder a perguntas sobre a oferta formativa do IPVC, recorrendo a tecnologias open source e a uma arquitetura baseada em RAG.

A solução utiliza:

- ficheiros CSV com informação estruturada;
- PDFs, brochuras e regulamentos com informação não estruturada;
- uma base vetorial local em ChromaDB;
- embeddings gerados com SentenceTransformers;
- um modelo de linguagem local executado através do Ollama.

## Funcionalidades principais

- Consulta de cursos por escola do IPVC.
- Consulta de cursos por localidade.
- Consulta de cursos por grau de ensino.
- Consulta de CTeSP, licenciaturas, mestrados e pós-graduações.
- Consulta de escolas do IPVC.
- Consulta de provas de ingresso.
- Consulta de médias e notas do último colocado.
- Consulta de saídas profissionais.
- Comparação entre cursos.
- Recomendação de cursos com base em interesses do utilizador.
- Recomendação de cursos considerando a média de acesso.
- Apresentação das fontes consultadas.
- Interface web simples desenvolvida em Streamlit.

## Dados utilizados

### Dados estruturados

- `cursos_ipvc.csv`
- `escolas_ipvc.csv`
- `medias_ipvc.csv`

### Dados não estruturados

- Brochuras do IPVC;
- regulamentos;
- documentos PDF locais.

## Tecnologias utilizadas

- Python
- Streamlit
- Pandas
- pypdf
- SentenceTransformers
- ChromaDB
- Ollama
- Llama 3


##
- Paulo Simões Nº31377
- Francisco Matos Nº31406