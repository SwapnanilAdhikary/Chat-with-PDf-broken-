import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import gc

# Load environment variables
load_dotenv()

# Set the Hugging Face token
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit
st.title("Q&A with PDF Documents")
st.write("Upload your PDF files and ask questions about their content.")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Process uploaded PDFs
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            try:
                temppdf = f"./temp_{uploaded_file.name}"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)
                st.write(f"Loaded {len(docs)} documents from {uploaded_file.name}.")
                
                # Clean up temporary file
                os.remove(temppdf)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        # Split and create embeddings for the documents
        if documents:
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                
                # Create vector store
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.write("Vector store created successfully.")

                retriever = vectorstore.as_retriever()

                # Q&A prompt
                system_prompt = (
                    "You are an assistant for answering questions based on the provided documents. "
                    "Use the context from the documents to provide concise and accurate answers."
                    "\n\n"
                    "{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                def get_session_history(session: str) -> BaseChatMessageHistory:
                    if session not in st.session_state.store:
                        st.session_state.store[session] = ChatMessageHistory()
                    return st.session_state.store[session]

                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain, get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )

                user_input = st.text_input("Your question:")
                if user_input:
                    session_history = get_session_history("default_session")
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": "default_session"}},
                        )
                        st.write("Assistant:", response['answer'])
                        st.write("Chat History:", session_history.messages)
                    except Exception as e:
                        st.error(f"Error processing your question: {e}")
            except Exception as e:
                st.error(f"Error during document processing: {e}")
                gc.collect()  # Trigger garbage collection
        else:
            st.warning("No valid documents found. Please upload a valid PDF.")
else:
    st.warning("Please enter the Groq API Key.")
