from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_cohere import CohereEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st 
import os,io,tempfile
# from dotenv import load_dotenv
# load_dotenv()

st.set_page_config(page_title="DocumentGPT", page_icon=":💬:", layout="wide")
st.header("DocumentGPT 💬")

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "RAG"

def get_pdf_text(file_path):
    loader = UnstructuredFileLoader(
        file_path=file_path)
    # loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

def get_txt_text(file_path):
    loader = TextLoader(file_path)
    splits = loader.load_and_split()
    return splits

with st.sidebar:

    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    if st.button("Process"):
        Documents = []
        
        if uploaded_files:
            st.write("Files Loaded Splitting...")
            for uploaded_file in uploaded_files:
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as fp:
                        fp.write(uploaded_file.read())
                        temp_file_path = fp.name

                    split_tup = os.path.splitext(uploaded_file.name)
                    file_extension = split_tup[1]

                    if file_extension == ".pdf":
                        Documents.extend(get_pdf_text(temp_file_path))

                    elif file_extension == ".txt":
                        Documents.extend(get_txt_text(temp_file_path))

                except Exception as e:
                    st.error(f"Error processing this file: {uploaded_file.name} {e}")
                finally:
                    os.remove(temp_file_path)
        else:
            st.error("No file uploaded.")

        if Documents:
            st.write("Indexing Please Wait...")
            # indexing part goes here
            try:
                url = os.getenv("cluster_url")
                api_key = os.getenv("gd_api_key")
                embeddings = HuggingFaceEmbeddings()
                # embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.getenv("cohere_api_key"))
                qdrant = Qdrant.from_documents(
                    Documents,
                    embeddings,
                    url=url,
                    prefer_grpc=True,
                    api_key=api_key,
                    force_recreate=True,
                    collection_name="my_documents")
                
                st.write("Indexing Done")
                st.session_state["processtrue"] = True
                if "langchain_messages" in st.session_state:
                    st.session_state["langchain_messages"] = []

            except Exception as e:
                st.error(f"Error indexing: {e}")

if "processtrue" in st.session_state:
    from chat import main
    main()

else:
    st.info("Please Upload Your Files.")