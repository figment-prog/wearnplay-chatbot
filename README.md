# wearnplay Customer Service Chatbot

This is an AI chatbot project for an internship, built with Python, Streamlit, and Langchain.

It acts as a customer service assistant for the T-shirt and hoodie brand "wearnplay". It uses a RAG (Retrieval-Augmented Generation) system with a local vector database (ChromaDB) to answer questions based *only* on the knowledge provided in the `faq.txt` file.

## Features

* **Graphical UI:** Built with Streamlit for a simple and clean web app interface.
* **Smart Retrieval:** Uses a Reranker model (`BAAI/bge-reranker-base`) to understand synonyms (like "destroyed" vs. "damaged") and find the most accurate answers.
* **Local LLM:** Runs using a local LLM (like `gemma3:1b`) through Ollama.

## How to Run This Project

**Warning:** This project does not use a virtual environment and will install packages globally.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/figment-prog/wearnplay-chatbot.git
cd wearnplay-chatbot
    ```

2.  ## 🛠️ Installation

This project uses several key Python libraries. You can install them all using pip:

1.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate.bat # Windows
    # source venv/bin/activate # Mac/Linux
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Required Libraries:

* **`streamlit`**: For creating the interactive web application UI. 🌐
* **`langchain`, `langchain-core`, `langchain-community`**: The core framework for building the AI logic and connecting components. 🧠
* **`langchain-text-splitters`**: For breaking down the FAQ document into smaller chunks. ✂️
* **`langchain-ollama`**: To connect with the local Ollama LLM (like Gemma). 🤖
* **`langchain-chroma`**: For interacting with the Chroma vector database. 💾
* **`langchain-huggingface`**: To load the embedding and reranker models from Hugging Face. 🤗
* **`chromadb`**: The vector database used to store and search document embeddings. 🔎
* **`sentence-transformers`**: Provides the embedding model (`all-MiniLM-L6-v2`) and the reranker model (`BAAI/bge-reranker-base`). ↔️
* **`pypdf`**: (Included just in case, though the current code uses `TextLoader`) For loading text from PDF documents. 📄

3.  **Run the Chatbot App:**
    * **Important:** Make sure your Ollama application is running and you have pulled the `gemma3:1b` model (`ollama pull gemma3:1b`).
    * Run the Streamlit app:
        ```bash
        streamlit run chatbot.py
        ```
