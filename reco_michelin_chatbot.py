#!/usr/bin/env python3
"""
Chatbot de Recommandation Michelin – Exemple complet (2025)
-----------------------------------------------------------
Ce script implémente un chatbot de recommandation de produits Michelin basé sur une
approche Retrieval‑Augmented Generation (RAG) de bout en bout.

Technologies clés
=================
* **LangChain ≥ 0.2** – orchestration
* **OpenAI API** – embeddings + génération (GPT‑4o‑mini par défaut)
* **FAISS** – base vectorielle locale haute performance
* **Streamlit ≥ 1.35** – interface web légère

Structure du projet
===================
project/
├── reco_michelin_chatbot.py   ← *ce fichier*
├── catalog.csv                ← catalogue produit (exemple ci‑dessous)
└── faiss_index/               ← généré automatiquement au premier lancement

Exemple de catalog.csv (UTF‑8)
------------------------------
```csv
id,name,description,category,price,link
PIL-PS5, Michelin Pilot Sport 5, Pneumatique été haute performance pour voitures sportives, Été, 179.90, https://...
CRO-CROSS, Michelin CrossClimate 2, Pneumatique toutes saisons avec excellente traction sur neige, Toutes‑saisons, 149.50, https://...
```

Installation
============
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install langchain faiss-cpu streamlit openai pandas python-dotenv
export OPENAI_API_KEY="sk-..."      # ou placer la clé dans un fichier .env
```

Utilisation
===========
```bash
streamlit run reco_michelin_chatbot.py
```
Puis ouvrez http://localhost:8501 et commencez à discuter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.callbacks import StreamlitCallbackHandler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()  # charge OPENAI_API_KEY (si présent) à partir du .env

CATALOG_PATH = Path("catalog.csv")  # CSV du catalogue produit
INDEX_DIR = Path("faiss_index")     # dossier de l'index vectoriel
EMBEDDING_MODEL = "text-embedding-3-large"  # modèle d'embeddings OpenAI
CHAT_MODEL = "gpt-4o-mini"                  # modèle de chat OpenAI
TOP_K = 4                                     # documents à récupérer
TEMPERATURE = 0.2                             # créativité des réponses

# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------

def build_vectorstore(catalog_path: Path, index_dir: Path) -> FAISS:
    """Crée et enregistre un index FAISS à partir du CSV de catalogue."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Fichier catalogue introuvable : {catalog_path}")

    st.info("📦 Construction de l'index vectoriel FAISS en cours …")

    df = pd.read_csv(catalog_path)
    docs: List[Document] = []
    for _, row in df.iterrows():
        content = f"{row['name']}. {row['description']}"
        meta = row.to_dict()
        docs.append(Document(page_content=content, metadata=meta))

    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embedder)
    vectorstore.save_local(str(index_dir))
    st.success("✅ Index créé avec succès !")
    return vectorstore


def load_vectorstore(index_dir: Path) -> FAISS:
    """Charge un index FAISS existant."""
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(str(index_dir), embedder, allow_dangerous_deserialization=True)


def get_rag_chain(vectorstore: FAISS) -> RetrievalQA:
    """Construit la chaîne RetrievalQA avec un prompt personnalisé."""
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=TEMPERATURE, streaming=True)

    prompt_template = (
        "Sei un esperto assistente Michelin. Utilizza CONTEXT per consigliare il miglior "
        "prodotto in risposta alla domanda dell'utente. Se non trovi corrispondenze, rispondi "
        '"Mi dispiace, non ho trovato un prodotto adeguato.".\n\n'
        "CONTEXT:\n{context}\n\n"  # sera rempli par RetrievalQA
        "DOMANDA: {question}\n\n"
        "***\nRisposta (max 150 parole, includi nome prodotto, motivazioni e link):"
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Chatbot Recommandation Michelin", page_icon="🚗", layout="wide")
st.title("🤖 Chatbot de Recommandation Michelin")
st.caption("Posez votre question en italien ou en anglais et recevez un conseil personnalisé sur les pneus Michelin.")

# Chargement ou création de l'index
if not INDEX_DIR.exists():
    vectorstore = build_vectorstore(CATALOG_PATH, INDEX_DIR)
else:
    vectorstore = load_vectorstore(INDEX_DIR)

rag_chain = get_rag_chain(vectorstore)

# Session de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.chat_message(message["role"]).markdown(message["content"])

user_prompt = st.chat_input("📨 Écrivez votre question ici…")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    st.chat_message("user").markdown(user_prompt)

    with st.chat_message("assistant"):
        callback_handler = StreamlitCallbackHandler(st.container())
        result = rag_chain(user_prompt, callbacks=[callback_handler])
        answer = result["result"]
        st.markdown(answer)

        # Développer pour afficher les sources
        with st.expander("Sources consultées"):
            for doc in result["source_documents"]:
                meta = doc.metadata
                st.markdown(f"**{meta.get('name', 'Produit inconnu')}** – Catégorie : {meta.get('category', '-')}")
                st.markdown(doc.page_content[:250] + "…")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# -----------------------------------------------------------------------------
# Mode CLI (optionnel, lancé depuis le terminal sans Streamlit)
# -----------------------------------------------------------------------------
if __name__ == "__main__" and not st.runtime.exists():
    # Mode test en ligne de commande
    print("=== Michelin Recommender CLI ===")
    if not INDEX_DIR.exists():
        vectorstore_cli = build_vectorstore(CATALOG_PATH, INDEX_DIR)
    else:
        vectorstore_cli = load_vectorstore(INDEX_DIR)
    chain_cli = get_rag_chain(vectorstore_cli)
    try:
        while True:
            q = input("Question (vide pour quitter): ").strip()
            if not q:
                break
            res = chain_cli(q)
            print("\n> ", res["result"], "\n")
    except KeyboardInterrupt:
        print("\nAu revoir!")
