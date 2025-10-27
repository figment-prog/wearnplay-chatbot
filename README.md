# wearnplay Customer Service Chatbot

This is an AI chatbot project for an internship, built with Python, Streamlit, and Langchain.

It acts as a customer service assistant for the T-shirt and hoodie brand "wearnplay". It uses a RAG (Retrieval-Augmented Generation) system with a local vector database (ChromaDB) to answer questions based *only* on the knowledge provided in the `faq.txt` file.

## Features ✨

* **Graphical UI:** Built with Streamlit for a simple and clean web app interface. 🌐
* **Smart Retrieval:** Uses a Reranker model (`BAAI/bge-reranker-base`) to understand synonyms (like "destroyed" vs. "damaged") and find the most accurate answers. 🧠
* **Local LLM:** Runs using a local LLM (like `gemma3:1b`) through Ollama. 🤖

---

## 🚀 How to Run This Project

### 1. Clone the Repository:

First, get the project files onto your computer.

```bash
git clone [https://github.com/figment-prog/wearnplay-chatbot.git](https://github.com/figment-prog/wearnplay-chatbot.git)
cd wearnplay-chatbot
2. Installation & Requirements 🛠️
Warning: The basic instructions below assume installation without a virtual environment, which will install packages globally. Using a virtual environment is strongly recommended for Python projects to avoid conflicts.

Recommended Setup (Using a Virtual Environment)
Create and Activate a Virtual Environment:

Bash

# Create the environment (only needs to be done once)
python -m venv venv

# Activate the environment (do this every time you open the project)
.\venv\Scripts\activate.bat # Windows Command Prompt
# source venv/bin/activate  # Mac/Linux Terminal
Install Dependencies within the venv:

Bash

pip install -r requirements.txt
Basic Setup (Global Installation - Not Recommended)
If you choose not to use a virtual environment, you can install the packages directly:

Install Dependencies Globally:

Bash

pip install -r requirements.txt
Required Libraries Explained:
streamlit: For creating the interactive web application UI. 🌐

langchain, langchain-core, langchain-community: The core framework for building the AI logic and connecting components. 🧠

langchain-text-splitters: For breaking down the FAQ document into smaller chunks. ✂️

langchain-ollama: To connect with the local Ollama LLM (like Gemma). 🤖

langchain-chroma: For interacting with the Chroma vector database. 💾

langchain-huggingface: To load the embedding and reranker models from Hugging Face. 🤗

chromadb: The vector database used to store and search document embeddings. 🔎

sentence-transformers: Provides the embedding model (all-MiniLM-L6-v2) and the reranker model (BAAI/bge-reranker-base). ↔️

pypdf: (Included for potential future use) Allows loading text from PDF documents. 📄

3. Run the Chatbot App ▶️
Important: Make sure your Ollama application is running on your computer.

Make sure you have pulled the required LLM model:

Bash

ollama pull gemma3:1b
Run the Streamlit app from your terminal (make sure your Python script is named chatbot.py):

Bash

streamlit run chatbot.py
Your web browser should open automatically to the chatbot interface!