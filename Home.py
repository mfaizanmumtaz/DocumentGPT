from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_cohere import CohereEmbeddings
import streamlit as st 
import os,uuid

st.set_page_config(page_title="DocumentGPT", page_icon=":💬:", layout="wide")
st.header("DocumentGPT 💬")

with st.sidebar:

    uploaded_files = st.file_uploader("Upload your file", type=['pdf','txt'], accept_multiple_files=True)

    if st.button("Process"):

        def get_pdf_text(file_path):
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            return pages

        def get_txt_text(file_path):
            loader = TextLoader(file_path)
            splits = loader.load_and_split()
            return splits

        def generate_unique_filename(directory, original_filename):
            if not os.path.exists(directory):
                os.mkdir(directory)
                
            base_name, file_extension = os.path.splitext(original_filename)
            unique_id = uuid.uuid4()
            unique_filename = f"{base_name}_{unique_id}{file_extension}"
            return os.path.join(directory, unique_filename)

        Documents = []
        
        if uploaded_files:
            st.write("Files Loaded Splitting And Chunking...")
            for uploaded_file in uploaded_files:
                split_tup = os.path.splitext(uploaded_file.name)
                file_path = generate_unique_filename("data",uploaded_file.name)
                try:
                    with open(file_path,'wb') as file:
                        file.write(uploaded_file.read())

                    file_extension = split_tup[1]
                    if file_extension == ".pdf":
                        Documents.extend(get_pdf_text(file_path))

                    elif file_extension == ".txt":
                        Documents.extend(get_txt_text(file_path))
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                finally:
                    os.remove(file_path)
            
            # Indexing part Goes Here
            st.write("Indexing Please Wait...")
            try:
                url = os.getenv("cluster_url")
                api_key = os.getenv("gd_api_key")
                embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.getenv("cohere_api_key"))
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
                st.error(f"Error indexing documents reload the page and try again: {e}")
        else:
            st.error("No file uploaded.")

if "processtrue" in st.session_state:
    from chat import main
    main()

else:
    st.info("Please Upload Your Files.")