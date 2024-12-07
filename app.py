from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.tools import Tool
import wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GEMINI_API_KEY"])

wikipedia = WikipediaAPIWrapper(lang="pt")

# Tool Wikipedia
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia.run,
    description=(
        "Use esta ferramenta **apenas** para buscar definições ou conceitos sobre "
        "direito e contratos com foco jurídico. Insira termos precisos e claros."
        "As respostas devem ser simples de entender para pessoas leigas sobre o assunto"))


# PDF Tool
def extract_pdf_content(file_path: str):
    """
    Função utilizada para extrair os dados mais relevantes de um contrato em PDF.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)
    except Exception as e:
        return f"Erro ao processar o arquivo PDF: {e}"

pdf_tool = Tool(
    name="pdf_extractor",
    func=extract_pdf_content,
    description="Use para obter dados de arquivos PDF e extrair informações jurídicas relevantes. A resposta deverá ser em português"
)



tools = [wikipedia_tool, pdf_tool]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Agente
zero_shot_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)


prompt_inicial = (
    "Você é um assistente jurídico especializado em contratos. "
    "Use informações disponíveis nas ferramentas ou em interações anteriores para responder. "
    "Todas as respostas devem estar em português e ser claras para leigos."
)

prompt_pdf = (
    "Você é um assistente jurídico especializado em contratos. "
    "Resuma e numere os pontos mais importantes do contrato com riqueza de detalhes "
    "Todas as respostas devem estar em português e ser claras para leigos."
)


st.set_page_config(page_title="Assistente Jurídico", layout="centered")

st.title("Assistente Jurídico com LangChain")
st.write("Interaja com o assistente para tirar dúvidas sobre contratos e buscar informações relevantes.")

st.subheader("Envie suas perguntas")
user_input = st.text_input("Digite sua dúvida ou termo jurídico:")

st.subheader("Envie um arquivo PDF para análise")
uploaded_file = st.file_uploader("Selecione um arquivo PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processando o arquivo PDF..."):
        try:
        
            DATA_FOLDER = "data"
            os.makedirs(DATA_FOLDER, exist_ok=True)
            file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            response = zero_shot_agent.invoke(f"{prompt_pdf} .Arquivo pdf: {file_path}")
            st.success("Processamente realizado com sucesso!")
            st.markdown("🤖 Análise do assistente virtual 🤖")
            st.markdown(f"""{response['output']}""")
            os.remove(file_path)
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")


if user_input:
    with st.spinner("Gerando resposta..."):
        try:
            response = zero_shot_agent.invoke(f"{prompt_inicial} {user_input}")
            st.success("Resposta gerada com sucesso!")
            st.markdown("🤖 Análise do assistente virtual 🤖")
            st.markdown(f"""{response['output']}""")
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")
