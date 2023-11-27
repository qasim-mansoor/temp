API_KEY = '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'
import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = ChatOpenAI(model_name='gpt-4', openai_api_key=API_KEY)
    return llm

st.title("Contract Analysis System")
st.markdown("<h3 style='text-align: center; color: white;'>Copyright ® AI Workbay</a></h3>", unsafe_allow_html=True)

def uploader_callback():
    st.session_state.pop("history", None)
    st.session_state.pop("generated", None)
    st.session_state.pop("past", None)
    
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="pdf", on_change=uploader_callback)
old_file = None



if uploaded_file:

    # if uploaded_file != old_file:
    #     st.session_state.pop("history", None)
    #     st.session_state.pop("generated", None)
    #     st.session_state.pop("past", None)

    #     old_file = uploaded_file

    # st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name]

    #use tempfile because PDFLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(file_path=tmp_file_path)
    data = loader.load_and_split()
    #st.json(data)
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

    chromaDB = Chroma(persist_directory=DB_FAISS_PATH)
    db = chromaDB.from_documents(data, embeddings)
    # db.save_local(DB_FAISS_PATH)
    # db.PersistentClient(path=DB_FAISS_PATH)
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey !"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_area("Query:", placeholder="Query your data here", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")



    

