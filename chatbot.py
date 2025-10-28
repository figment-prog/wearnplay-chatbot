import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

@st.cache_resource
def load_retrieval_chain():
    print("--- ⏳ Loading models and building chain... (This will run only once) ---")

    llm = Ollama(model="gemma3:1b")
    prompt_template = """
    You are a helpful and polite customer service assistant for 'wearnplay', a T-shirt and hoodie company.
Answer the user's question based *only* on the context provided below.
Be concise and directly answer the question. Do not add conversational fluff unless asked.

If the context doesn't contain the answer to the user's specific question:
1.  Politely say you don't have the exact information in your knowledge base.
2.  Suggest they contact customer support directly for further assistance at support@wearnplay.com.
3.  Do not make up an answer or search outside the provided context.
   
    Context:
    {context}

    Question:
    {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    loader = TextLoader("faq.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    
    print("--- ✅ Chain is ready! ---")
    return retrieval_chain


st.title("🤖 wearnplay Customer Service")
st.caption("Ask me anything about our T-shirts, hoodies, and policies!")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with your wearnplay order today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

try:
    retrieval_chain = load_retrieval_chain()
except Exception as e:
    st.error(f"Failed to load the chatbot. Is Ollama running? Error: {e}")
    st.stop()


if user_prompt := st.chat_input("What is your question?"):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        with st.spinner("⏳ Thinking..."):
            try:
                response = retrieval_chain.invoke({"input": user_prompt})
                bot_response = response['answer']
            except Exception as e:
                bot_response = f"I'm sorry, an error occurred: {e}"
        
        st.markdown(bot_response)
    
    st.session_state.messages.append({"role": "assistant", "content": bot_response})