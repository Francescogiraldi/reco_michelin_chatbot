# Chatbot de Recommandation Michelin

Bienvenue sur **Chatbot Recommandation Michelin**, un assistant conversationnel basé sur une approche *Retrieval‑Augmented Generation* (RAG) qui aide les clients à choisir le pneu Michelin le plus adapté à leurs besoins.


---

## ✨ Fonctionnalités

* **Recherche sémantique** dans votre catalogue produit via un index vectoriel FAISS
* **Génération de réponses** contextuelles avec GPT‑4o‑mini (OpenAI)
* Interface **web** avec Streamlit et **CLI** pour les tests rapides
* Index construit localement à partir de votre `catalog.csv` — aucune donnée sensible n’est envoyée vers un service tiers hors OpenAI
* Déploiement possible sur un simple serveur, Docker ou n’importe quel PaaS (Railway, Render, etc.)

---

## 🗂️ Structure du projet

```text
project/
├── reco_michelin_chatbot.py   # script principal
├── catalog.csv                # votre catalogue produit (UTF‑8)
└── faiss_index/               # dossier de l’index (généré au 1ᵉʳ lancement)
```

Le fichier **`catalog.csv`** doit contenir au minimum :

| Colonne       | Description                                  | Exemple                              |
| ------------- | -------------------------------------------- | ------------------------------------ |
| `id`          | Identifiant unique du produit                | `PIL-PS5`                            |
| `name`        | Nom court                                    | `Michelin Pilot Sport 5`             |
| `description` | Description marketing ou technique           | `Pneumatique été haute performance…` |
| `category`    | Catégorie (Été, Hiver, Toutes‑saisons, etc.) | `Été`                                |
| `price`       | Prix public TTC                              | `179.90`                             |
| `link`        | URL vers la fiche produit                    | `https://…`                          |

---

## 🚀 Installation rapide

1. **Cloner** le dépôt :

   ```bash
   git clone https://github.com/votre-org/reco-michelin-chatbot.git
   cd reco-michelin-chatbot
   ```

2. **Créer un environnement virtuel** et installer les dépendances :

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt  # ou utiliser la commande ci‑dessous
   # pip install langchain faiss-cpu streamlit openai pandas python-dotenv
   ```

3. **Configurer** la clé API OpenAI :

   ```bash
   export OPENAI_API_KEY="sk-…"      # ou créer un fichier .env
   ```

---

## 🏃 Lancement

### Interface web (Streamlit)

```bash
streamlit run reco_michelin_chatbot.py
```

Ouvrez ensuite [http://localhost:8501](http://localhost:8501) dans votre navigateur préféré.

### Ligne de commande

```bash
python reco_michelin_chatbot.py  # Ctrl+C pour quitter
```

---

## ⚙️ Variables d’environnement

| Variable          | Description                 | Défaut                   |
| ----------------- | --------------------------- | ------------------------ |
| `OPENAI_API_KEY`  | Clé API OpenAI              | *obligatoire*            |
| `EMBEDDING_MODEL` | Modèle d’embeddings (texte) | `text-embedding-3-large` |
| `CHAT_MODEL`      | Modèle de chat              | `gpt-4o-mini`            |

Vous pouvez surcharger ces valeurs dans un fichier **`.env`** à la racine du projet.

---

## 🛠️ Personnalisation

* **Prompt** – Rendez‑vous ligne `prompt_template` dans le script pour adapter la tonalité et la longueur des réponses.
* **TOP\_K** – Nombre de documents récupérés.
* **catalog.csv** – Ajoutez ou modifiez vos propres colonnes (par ex. `season`, `width`, `diameter`).

---

## 📦 Déploiement sur Docker (exemple)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install langchain faiss-cpu streamlit openai pandas python-dotenv

EXPOSE 8501
CMD ["streamlit", "run", "reco_michelin_chatbot.py", "--server.address", "0.0.0.0"]
```

Construisez et lancez :

```bash
docker build -t michelin-chatbot .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-… michelin-chatbot
```

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! Ouvrez une *issue* ou envoyez une *pull request*.

---

## 📝 Licence

Distribué sous licence **MIT**. Consultez le fichier `LICENSE` pour plus d’informations.
