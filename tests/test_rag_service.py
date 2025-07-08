"""Tests for the RAG service module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
from langchain.schema import Document

from src.services.rag_service import RAGService
from src.config import Settings


class TestRAGService:
    """Test cases for RAGService."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.openai_api_key = "test-key"
        settings.embedding_model = "text-embedding-3-small"
        settings.chat_model = "gpt-4o-mini"
        settings.temperature = 0.2
        settings.top_k = 4
        settings.catalog_path = Path("test_catalog.csv")
        settings.index_path = Path("test_index")
        settings.language = "en"
        return settings
    
    @pytest.fixture
    def sample_catalog_data(self):
        """Create sample catalog data for testing."""
        return pd.DataFrame([
            {
                "id": "PIL-PS5",
                "name": "Michelin Pilot Sport 5",
                "description": "High performance summer tire",
                "category": "Summer",
                "price": 179.90,
                "link": "https://example.com/ps5"
            },
            {
                "id": "CRO-CROSS",
                "name": "Michelin CrossClimate 2",
                "description": "All-season tire with excellent snow traction",
                "category": "All-Season",
                "price": 149.50,
                "link": "https://example.com/crossclimate"
            }
        ])
    
    def test_initialization(self, mock_settings):
        """Test RAGService initialization."""
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'):
            service = RAGService(mock_settings)
            assert service.settings == mock_settings
            assert service.embeddings is not None
            assert service.llm is not None
    
    @patch('pandas.read_csv')
    def test_load_catalog_success(self, mock_read_csv, mock_settings, sample_catalog_data):
        """Test successful catalog loading."""
        mock_read_csv.return_value = sample_catalog_data
        mock_settings.catalog_path.exists.return_value = True
        
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'):
            service = RAGService(mock_settings)
            catalog = service._load_catalog()
            
            assert len(catalog.products) == 2
            assert catalog.products[0].id == "PIL-PS5"
            assert catalog.products[1].id == "CRO-CROSS"
    
    def test_load_catalog_file_not_found(self, mock_settings):
        """Test catalog loading when file doesn't exist."""
        mock_settings.catalog_path.exists.return_value = False
        
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'):
            service = RAGService(mock_settings)
            
            with pytest.raises(FileNotFoundError):
                service._load_catalog()
    
    @patch('src.services.rag_service.FAISS')
    def test_build_vectorstore(self, mock_faiss, mock_settings, sample_catalog_data):
        """Test vector store building."""
        mock_vectorstore = Mock()
        mock_faiss.from_documents.return_value = mock_vectorstore
        
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, '_load_catalog') as mock_load:
            
            # Mock catalog
            mock_catalog = Mock()
            mock_catalog.products = [
                Mock(to_search_content=lambda: "Pilot Sport 5 content"),
                Mock(to_search_content=lambda: "CrossClimate 2 content")
            ]
            mock_load.return_value = mock_catalog
            
            service = RAGService(mock_settings)
            vectorstore = service._build_vectorstore()
            
            assert vectorstore == mock_vectorstore
            mock_faiss.from_documents.assert_called_once()
            mock_vectorstore.save_local.assert_called_once()
    
    @patch('src.services.rag_service.FAISS')
    def test_load_vectorstore(self, mock_faiss, mock_settings):
        """Test vector store loading."""
        mock_vectorstore = Mock()
        mock_faiss.load_local.return_value = mock_vectorstore
        mock_settings.index_path.exists.return_value = True
        
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'):
            service = RAGService(mock_settings)
            vectorstore = service._load_vectorstore()
            
            assert vectorstore == mock_vectorstore
            mock_faiss.load_local.assert_called_once()
    
    def test_load_vectorstore_not_found(self, mock_settings):
        """Test vector store loading when index doesn't exist."""
        mock_settings.index_path.exists.return_value = False
        
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'):
            service = RAGService(mock_settings)
            
            with pytest.raises(FileNotFoundError):
                service._load_vectorstore()
    
    def test_get_vectorstore_build_new(self, mock_settings):
        """Test getting vector store when building new one."""
        mock_settings.index_path.exists.return_value = False
        
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, '_build_vectorstore') as mock_build:
            
            mock_vectorstore = Mock()
            mock_build.return_value = mock_vectorstore
            
            service = RAGService(mock_settings)
            vectorstore = service.get_vectorstore()
            
            assert vectorstore == mock_vectorstore
            mock_build.assert_called_once()
    
    def test_get_vectorstore_load_existing(self, mock_settings):
        """Test getting vector store when loading existing one."""
        mock_settings.index_path.exists.return_value = True
        
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, '_load_vectorstore') as mock_load:
            
            mock_vectorstore = Mock()
            mock_load.return_value = mock_vectorstore
            
            service = RAGService(mock_settings)
            vectorstore = service.get_vectorstore()
            
            assert vectorstore == mock_vectorstore
            mock_load.assert_called_once()
    
    def test_create_rag_chain(self, mock_settings):
        """Test RAG chain creation."""
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch('src.services.rag_service.RetrievalQA') as mock_qa:
            
            mock_chain = Mock()
            mock_qa.from_chain_type.return_value = mock_chain
            
            service = RAGService(mock_settings)
            chain = service._create_rag_chain(mock_vectorstore)
            
            assert chain == mock_chain
            mock_qa.from_chain_type.assert_called_once()
    
    def test_query_success(self, mock_settings):
        """Test successful query execution."""
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, 'get_vectorstore') as mock_get_vs, \
             patch.object(RAGService, '_create_rag_chain') as mock_create_chain:
            
            # Mock chain response
            mock_chain = Mock()
            mock_chain.return_value = {
                "result": "Test response",
                "source_documents": [Mock()]
            }
            mock_create_chain.return_value = mock_chain
            
            service = RAGService(mock_settings)
            response = service.query("test question")
            
            assert response["result"] == "Test response"
            assert "source_documents" in response
    
    def test_query_error_handling(self, mock_settings):
        """Test query error handling."""
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, 'get_vectorstore') as mock_get_vs, \
             patch.object(RAGService, '_create_rag_chain') as mock_create_chain:
            
            # Mock chain to raise exception
            mock_chain = Mock()
            mock_chain.side_effect = Exception("Test error")
            mock_create_chain.return_value = mock_chain
            
            service = RAGService(mock_settings)
            
            with pytest.raises(Exception):
                service.query("test question")
    
    def test_get_similar_products(self, mock_settings):
        """Test getting similar products."""
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, 'get_vectorstore') as mock_get_vs:
            
            # Mock vectorstore search
            mock_vectorstore = Mock()
            mock_doc = Mock()
            mock_doc.metadata = {"id": "test-id", "name": "Test Product"}
            mock_vectorstore.similarity_search.return_value = [mock_doc]
            mock_get_vs.return_value = mock_vectorstore
            
            service = RAGService(mock_settings)
            products = service.get_similar_products("test query")
            
            assert len(products) == 1
            assert products[0].metadata["id"] == "test-id"
    
    def test_health_check_success(self, mock_settings):
        """Test successful health check."""
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, 'get_vectorstore') as mock_get_vs, \
             patch.object(RAGService, '_load_catalog') as mock_load_catalog:
            
            mock_catalog = Mock()
            mock_catalog.products = [Mock(), Mock()]  # 2 products
            mock_load_catalog.return_value = mock_catalog
            
            service = RAGService(mock_settings)
            status = service.health_check()
            
            assert status["status"] == "healthy"
            assert status["catalog_products"] == 2
            assert "vectorstore_ready" in status
    
    def test_health_check_failure(self, mock_settings):
        """Test health check with failure."""
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, 'get_vectorstore') as mock_get_vs, \
             patch.object(RAGService, '_load_catalog') as mock_load_catalog:
            
            mock_load_catalog.side_effect = Exception("Catalog error")
            
            service = RAGService(mock_settings)
            status = service.health_check()
            
            assert status["status"] == "unhealthy"
            assert "error" in status
    
    def test_rebuild_index(self, mock_settings):
        """Test index rebuilding."""
        with patch('src.services.rag_service.OpenAIEmbeddings'), \
             patch('src.services.rag_service.ChatOpenAI'), \
             patch.object(RAGService, '_build_vectorstore') as mock_build, \
             patch('shutil.rmtree') as mock_rmtree:
            
            mock_settings.index_path.exists.return_value = True
            mock_vectorstore = Mock()
            mock_build.return_value = mock_vectorstore
            
            service = RAGService(mock_settings)
            result = service.rebuild_index()
            
            assert result is True
            mock_rmtree.assert_called_once()
            mock_build.assert_called_once()