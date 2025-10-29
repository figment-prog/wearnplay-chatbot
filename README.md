# wearnplay Customer Service Chatbot

This is an AI chatbot project for an internship, built with Python, Streamlit, and Langchain.

It acts as a customer service assistant for the T-shirt and hoodie brand "wearnplay". It uses a RAG (Retrieval-Augmented Generation) system with a local vector database (ChromaDB) to answer questions based *only* on the knowledge provided in the `faq.txt` file.

## Features

* **Graphical UI:** Built with Streamlit for a simple and clean web app interface.
* **Smart Retrieval:** Uses a Reranker model (`BAAI/bge-reranker-base`) to understand synonyms (like "destroyed" vs. "damaged") and find the most accurate answers.
* **Local LLM:** Runs using a local LLM (like `gemma3:1b`) through Ollama.

## How to Run This Project

**Warning:** This project does not use a virtual environment and will install packages globally.
             You will need to install Ollama and pull the gemma3:1b model.
             ```
             ollama pull gemma3:1b
             ```

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/figment-prog/wearnplay-chatbot.git
    cd wearnplay-chatbot
    ```

2.  **Install the Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Chatbot App:**
    * **Important:** Make sure your Ollama application is running and you have pulled the `gemma3:1b` model (`ollama pull gemma3:1b`).
    * Run the Streamlit app:
        ```bash
        streamlit run chatbot.py
        ```
