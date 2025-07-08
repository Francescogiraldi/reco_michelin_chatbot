"""RAG (Retrieval-Augmented Generation) service for Michelin Chatbot."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from ..config import get_settings
from ..models.tire import TireProduct, TireCatalog
from ..utils.logging import LoggerMixin


class RAGService(LoggerMixin):
    """Service for handling RAG operations."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.settings = get_settings()
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize OpenAI components."""
        try:
            # Check if we're in demo mode
            if self.settings.openai_api_key == "demo_key":
                self.logger.warning("Running in DEMO MODE - OpenAI features disabled")
                self.embeddings = None
                self.llm = None
                return
            
            # Set OpenAI API key
            os.environ["OPENAI_API_KEY"] = self.settings.openai_api_key
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                openai_api_key=self.settings.openai_api_key
            )
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model_name=self.settings.chat_model,
                temperature=self.settings.temperature,
                openai_api_key=self.settings.openai_api_key,
                streaming=True
            )
            
            self.logger.info("RAG components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG components: {e}")
            # In case of error, fall back to demo mode
            self.logger.warning("Falling back to DEMO MODE")
            self.embeddings = None
            self.llm = None
    
    def load_catalog(self, catalog_path: Optional[Path] = None) -> TireCatalog:
        """Load and validate tire catalog from CSV.
        
        Args:
            catalog_path: Path to catalog CSV file
            
        Returns:
            Validated tire catalog
            
        Raises:
            FileNotFoundError: If catalog file doesn't exist
            ValueError: If catalog data is invalid
        """
        path = catalog_path or self.settings.catalog_path
        
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")
        
        try:
            self.logger.info(f"Loading catalog from: {path}")
            df = pd.read_csv(path, encoding='utf-8')
            
            # Validate required columns
            required_columns = {'id', 'name', 'description', 'category', 'price', 'link'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert to TireProduct objects
            products = []
            for _, row in df.iterrows():
                try:
                    product = TireProduct(**row.to_dict())
                    products.append(product)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid product row {row.get('id', 'unknown')}: {e}")
            
            catalog = TireCatalog(products=products, total_count=len(products))
            self.logger.info(f"Loaded {len(products)} products from catalog")
            
            return catalog
            
        except Exception as e:
            self.logger.error(f"Failed to load catalog: {e}")
            raise
    
    def build_vectorstore(self, catalog: TireCatalog, force_rebuild: bool = False) -> Optional[FAISS]:
        """Build or load FAISS vector store.
        
        Args:
            catalog: Tire catalog to index
            force_rebuild: Whether to force rebuild existing index
            
        Returns:
            FAISS vector store or None in demo mode
        """
        # Handle demo mode
        if self.embeddings is None:
            self.logger.info("Demo mode: Skipping vector store initialization")
            return None
            
        index_path = self.settings.index_dir
        
        # Check if index exists and force_rebuild is False
        if index_path.exists() and not force_rebuild:
            try:
                self.logger.info(f"Loading existing vector store from: {index_path}")
                vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.vectorstore = vectorstore
                return vectorstore
            except Exception as e:
                self.logger.warning(f"Failed to load existing index: {e}. Rebuilding...")
        
        # Build new index
        self.logger.info("Building new vector store...")
        
        try:
            # Create documents from catalog
            documents = []
            for product in catalog.products:
                content = product.to_search_content()
                metadata = product.to_dict()
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            # Create FAISS index
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Save index
            index_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(index_path))
            
            self.vectorstore = vectorstore
            self.logger.info(f"Vector store built and saved to: {index_path}")
            
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Failed to build vector store: {e}")
            raise
    
    def create_qa_chain(self, language: str = "fr") -> Optional[RetrievalQA]:
        """Create RetrievalQA chain with custom prompt.
        
        Args:
            language: Response language (fr, en, it, es, de)
            
        Returns:
            Configured RetrievalQA chain or None in demo mode
        """
        # Handle demo mode
        if self.embeddings is None or self.llm is None:
            self.logger.info("Demo mode: Skipping QA chain creation")
            return None
            
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call build_vectorstore first.")
        
        # Language-specific prompts
        prompts = {
            "fr": {
                "template": (
                    "Vous êtes un expert assistant Michelin. Utilisez le CONTEXTE pour recommander "
                    "le meilleur produit en réponse à la question de l'utilisateur. Si vous ne trouvez "
                    "pas de correspondance appropriée, répondez 'Je suis désolé, je n'ai pas trouvé "
                    "de produit adapté à votre demande.'\n\n"
                    "CONTEXTE:\n{context}\n\n"
                    "QUESTION: {question}\n\n"
                    "Réponse (max 150 mots, incluez le nom du produit, les raisons et le lien):"
                )
            },
            "en": {
                "template": (
                    "You are an expert Michelin assistant. Use the CONTEXT to recommend the best "
                    "product in response to the user's question. If you don't find an appropriate "
                    "match, respond 'I'm sorry, I couldn't find a suitable product for your needs.'\n\n"
                    "CONTEXT:\n{context}\n\n"
                    "QUESTION: {question}\n\n"
                    "Response (max 150 words, include product name, reasons, and link):"
                )
            },
            "it": {
                "template": (
                    "Sei un esperto assistente Michelin. Utilizza il CONTESTO per consigliare il miglior "
                    "prodotto in risposta alla domanda dell'utente. Se non trovi corrispondenze appropriate, "
                    "rispondi 'Mi dispiace, non ho trovato un prodotto adeguato alle tue esigenze.'\n\n"
                    "CONTESTO:\n{context}\n\n"
                    "DOMANDA: {question}\n\n"
                    "Risposta (max 150 parole, includi nome prodotto, motivazioni e link):"
                )
            }
        }
        
        # Get prompt template for language (default to French)
        prompt_config = prompts.get(language, prompts["fr"])
        prompt = ChatPromptTemplate.from_template(prompt_config["template"])
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.settings.top_k}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        self.qa_chain = qa_chain
        self.logger.info(f"QA chain created for language: {language}")
        
        return qa_chain
    
    def query(
        self,
        question: str,
        language: str = "fr",
        callbacks: Optional[List[BaseCallbackHandler]] = None
    ) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            question: User question
            language: Response language
            callbacks: Optional callback handlers
            
        Returns:
            Dictionary with result and source documents
        """
        # Handle demo mode
        if self.embeddings is None or self.llm is None:
            self.logger.info("Running in demo mode - returning mock response")
            demo_responses = {
                "fr": "🚗 **Mode Démo Activé** 🚗\n\nJe suis actuellement en mode démonstration. Pour utiliser pleinement le chatbot Michelin avec des recommandations personnalisées, veuillez configurer une clé API OpenAI valide dans le fichier .env.\n\n**Fonctionnalités disponibles en mode démo:**\n- Interface utilisateur complète\n- Navigation dans le catalogue\n- Aperçu des fonctionnalités\n\n**Pour activer le mode complet:**\n1. Obtenez une clé API OpenAI sur https://platform.openai.com\n2. Remplacez 'demo_key' par votre clé dans le fichier .env\n3. Redémarrez l'application",
                "en": "🚗 **Demo Mode Active** 🚗\n\nI'm currently running in demonstration mode. To use the full Michelin chatbot with personalized recommendations, please configure a valid OpenAI API key in the .env file.\n\n**Available features in demo mode:**\n- Complete user interface\n- Catalog browsing\n- Feature preview\n\n**To activate full mode:**\n1. Get an OpenAI API key from https://platform.openai.com\n2. Replace 'demo_key' with your key in the .env file\n3. Restart the application"
            }
            
            return {
                "result": demo_responses.get(language, demo_responses["en"]),
                "source_documents": [],
                "question": question,
                "language": language
            }
        
        if not self.qa_chain:
            self.create_qa_chain(language)
        
        try:
            self.logger.info(f"Processing query: {question[:50]}...")
            
            # Execute query
            result = self.qa_chain(
                {"query": question},
                callbacks=callbacks or []
            )
            
            # Log successful query
            self.logger.info("Query processed successfully")
            
            return {
                "result": result["result"],
                "source_documents": result["source_documents"],
                "question": question,
                "language": language
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            raise
    
    def get_similar_products(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Get similar products with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        k = k or self.settings.top_k
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            self.logger.info(f"Found {len(results)} similar products")
            return results
        except Exception as e:
            self.logger.error(f"Failed to search similar products: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on RAG service.
        
        Returns:
            Health status information
        """
        status = {
            "embeddings_initialized": self.embeddings is not None,
            "llm_initialized": self.llm is not None,
            "vectorstore_loaded": self.vectorstore is not None,
            "qa_chain_ready": self.qa_chain is not None,
            "index_exists": self.settings.index_dir.exists()
        }
        
        status["overall_healthy"] = all([
            status["embeddings_initialized"],
            status["llm_initialized"],
            status["vectorstore_loaded"]
        ])
        
        return status