import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("new")

# Load the embedding model with OpenAI API key
embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model="gpt-3.5-turbo")

# Setting up the PDF document loader
pdf_loader = PyPDFLoader(file_path='2404.07143v1.pdf')

# Load the PDF document
pdf_data = pdf_loader.load()

# Setting up a connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Define retrieval function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) 

# Converting Chroma db_connection to a Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    Your task is to provide assistance based on the context given by the user. 
    Make sure your answers are relevant and helpful."""),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:")
])

# Initialize chat model with OpenAI API key
chat_model = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Initialize output parser
output_parser = StrOutputParser()

# Define RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
st.title("✨ RAG System ✨")
st.subheader("Develop an advanced contextual question-answering AI system, guided by the principles of the 'Leave No Context Behind' paper.")

question = st.text_input("🤔 Please write your question:")

if st.button("Generate Answer 📣"):
    if question:
        response = rag_chain.invoke(question)
        st.markdown("📝📣 **Answer:** ✨", unsafe_allow_html=True)
        st.markdown(f"<span style='color: green'>{response}</span>", unsafe_allow_html=True)
    else:
        st.warning("📑💡 Please retry again.")
    st.balloons() 
    
    st.markdown("Thank you for exploring the RAG System! Feel free to ask any questions or provide feedback.", unsafe_allow_html=True)
