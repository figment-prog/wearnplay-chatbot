import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # This is your "engine"
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# --- 1. SET UP THE MODEL & PROMPT ---
print("✅ 1. Connecting to Ollama (gemma3:1b)...")
llm = Ollama(model="gemma3:1b")

prompt_template = """
You are a helpful customer service assistant for 'wearnplay', a T-shirt and hoodie company.
Answer the user's question based *only* on the context provided.
If the context doesn't contain the answer, just say "I'm sorry, I don't have that information."

Context:
{context}

Question:
{input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
print("   ...LLM and prompt are ready.")


# --- 2. LOAD AND PROCESS YOUR DOCUMENTS ---
print("✅ 2. Loading faq.txt...")
try:
    loader = TextLoader("faq.txt", encoding="utf-8")
    docs = loader.load()
except Exception as e:
    print(f"🔴 ERROR: Could not load faq.txt. Make sure it is in the same folder as app.py")
    print(f"Details: {e}")
    exit()

print("   ...Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

print("   ...Loading embedding model (this may take a minute on first run)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("   ...Adding documents to ChromaDB vector store...")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
print("   ...Vector store is ready.")


# --- 3. CREATE THE *SMART* RAG CHAIN ---
print("✅ 3. Creating the SMART RAG chain with Reranker...")

# 3a. Create the base retriever (to get 10 docs)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 3b. Initialize the Reranker model (the "engine")
print("   ...Loading Reranker model...")
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# --- 2. ADD THIS NEW LINE ---
# Now, create the compressor (the "car") and put the model inside it
compressor = CrossEncoderReranker(model=model, top_n=3) # Will keep the top 3 results
print("   ...Reranker model loaded.")


# 3c. Create the new "Contextual Compression Retriever"
# --- 3. CHANGE THIS LINE ---
# Pass the "compressor" object, not the "model" object
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, # <-- This is the fix
    base_retriever=base_retriever
)
print("   ...Smart retriever is ready.")

# 3d. Create the main chains (same as before)
document_chain = create_stuff_documents_chain(llm, prompt)

# 3e. Create the final chain
retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
print("   ...Chain is ready.")


# --- 4. RUN THE CHATBOT ---
print("\n--- 🤖 wearnplay Chatbot is ready! Ask a question (type 'exit' to quit) ---")
while True:
    try:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        
        print("Bot: ⏳...thinking...")
        
        response = retrieval_chain.invoke({"input": query})
        
        print(f"Bot: {response['answer']}\n")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break