"""Tests for the Streamlit application."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from pathlib import Path

# Note: These tests are designed to test the logic of the Streamlit app
# without actually running the Streamlit server


class TestStreamlitApp:
    """Test cases for Streamlit application logic."""
    
    @pytest.fixture
    def mock_rag_service(self):
        """Create a mock RAG service for testing."""
        service = Mock()
        service.health_check.return_value = {
            "status": "healthy",
            "catalog_products": 10,
            "vectorstore_ready": True
        }
        service.query.return_value = {
            "result": "Test response",
            "source_documents": [
                Mock(metadata={"name": "Test Product", "category": "Test"})
            ]
        }
        return service
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.app_title = "Test Michelin Chatbot"
        settings.app_description = "Test description"
        settings.language = "en"
        settings.max_chat_history = 50
        settings.streamlit_host = "localhost"
        settings.streamlit_port = 8501
        return settings
    
    def test_initialize_session_state(self):
        """Test session state initialization."""
        # Mock streamlit session state
        with patch('streamlit.session_state', {}) as mock_session:
            from src.ui.streamlit_app import initialize_session_state
            
            initialize_session_state()
            
            # Check that required keys are initialized
            expected_keys = [
                'messages', 'rag_service', 'show_sources', 
                'language', 'system_status'
            ]
            
            for key in expected_keys:
                assert key in mock_session
    
    @patch('src.ui.streamlit_app.get_settings')
    @patch('src.ui.streamlit_app.RAGService')
    def test_get_rag_service_cached(self, mock_rag_class, mock_get_settings, mock_settings, mock_rag_service):
        """Test RAG service caching."""
        mock_get_settings.return_value = mock_settings
        mock_rag_class.return_value = mock_rag_service
        
        with patch('streamlit.session_state', {'rag_service': None}) as mock_session:
            from src.ui.streamlit_app import get_rag_service
            
            # First call should create service
            service1 = get_rag_service()
            assert service1 == mock_rag_service
            
            # Second call should return cached service
            mock_session['rag_service'] = mock_rag_service
            service2 = get_rag_service()
            assert service2 == mock_rag_service
            
            # RAGService should only be called once
            mock_rag_class.assert_called_once()
    
    def test_format_message_user(self):
        """Test user message formatting."""
        from src.ui.streamlit_app import format_message
        
        message = {"role": "user", "content": "Test question"}
        
        with patch('streamlit.chat_message') as mock_chat:
            format_message(message)
            mock_chat.assert_called_with("user")
    
    def test_format_message_assistant(self):
        """Test assistant message formatting."""
        from src.ui.streamlit_app import format_message
        
        message = {
            "role": "assistant", 
            "content": "Test response",
            "sources": [Mock(metadata={"name": "Test Product"})]
        }
        
        with patch('streamlit.chat_message') as mock_chat, \
             patch('streamlit.session_state', {'show_sources': True}):
            format_message(message)
            mock_chat.assert_called_with("assistant")
    
    @patch('src.ui.streamlit_app.get_rag_service')
    def test_handle_user_input_success(self, mock_get_service, mock_rag_service):
        """Test successful user input handling."""
        mock_get_service.return_value = mock_rag_service
        
        with patch('streamlit.session_state', {'messages': []}) as mock_session:
            from src.ui.streamlit_app import handle_user_input
            
            handle_user_input("Test question")
            
            # Check that messages were added
            assert len(mock_session['messages']) == 2
            assert mock_session['messages'][0]['role'] == 'user'
            assert mock_session['messages'][1]['role'] == 'assistant'
    
    @patch('src.ui.streamlit_app.get_rag_service')
    def test_handle_user_input_error(self, mock_get_service):
        """Test user input handling with error."""
        mock_service = Mock()
        mock_service.query.side_effect = Exception("Test error")
        mock_get_service.return_value = mock_service
        
        with patch('streamlit.session_state', {'messages': []}) as mock_session, \
             patch('streamlit.error') as mock_error:
            from src.ui.streamlit_app import handle_user_input
            
            handle_user_input("Test question")
            
            # Check that error was displayed
            mock_error.assert_called()
    
    def test_render_sidebar_settings(self):
        """Test sidebar settings rendering."""
        with patch('streamlit.sidebar') as mock_sidebar, \
             patch('streamlit.session_state', {'language': 'en', 'show_sources': True}):
            from src.ui.streamlit_app import render_sidebar
            
            render_sidebar()
            
            # Verify sidebar was used
            mock_sidebar.header.assert_called()
    
    @patch('src.ui.streamlit_app.get_rag_service')
    def test_render_system_status_healthy(self, mock_get_service, mock_rag_service):
        """Test system status rendering when healthy."""
        mock_get_service.return_value = mock_rag_service
        
        with patch('streamlit.sidebar') as mock_sidebar:
            from src.ui.streamlit_app import render_system_status
            
            render_system_status()
            
            # Verify status was displayed
            mock_sidebar.success.assert_called()
    
    @patch('src.ui.streamlit_app.get_rag_service')
    def test_render_system_status_unhealthy(self, mock_get_service):
        """Test system status rendering when unhealthy."""
        mock_service = Mock()
        mock_service.health_check.return_value = {
            "status": "unhealthy",
            "error": "Test error"
        }
        mock_get_service.return_value = mock_service
        
        with patch('streamlit.sidebar') as mock_sidebar:
            from src.ui.streamlit_app import render_system_status
            
            render_system_status()
            
            # Verify error was displayed
            mock_sidebar.error.assert_called()
    
    def test_clear_chat_history(self):
        """Test chat history clearing."""
        with patch('streamlit.session_state', {'messages': ['msg1', 'msg2']}) as mock_session:
            from src.ui.streamlit_app import clear_chat_history
            
            clear_chat_history()
            
            assert len(mock_session['messages']) == 0
    
    def test_get_example_questions_english(self):
        """Test example questions in English."""
        from src.ui.streamlit_app import get_example_questions
        
        questions = get_example_questions('en')
        
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)
    
    def test_get_example_questions_italian(self):
        """Test example questions in Italian."""
        from src.ui.streamlit_app import get_example_questions
        
        questions = get_example_questions('it')
        
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)
    
    def test_get_example_questions_fallback(self):
        """Test example questions fallback for unknown language."""
        from src.ui.streamlit_app import get_example_questions
        
        questions = get_example_questions('unknown')
        
        # Should fallback to English
        assert isinstance(questions, list)
        assert len(questions) > 0
    
    @patch('streamlit.set_page_config')
    @patch('src.ui.streamlit_app.setup_logging')
    @patch('src.ui.streamlit_app.initialize_session_state')
    @patch('src.ui.streamlit_app.render_header')
    @patch('src.ui.streamlit_app.render_sidebar')
    @patch('src.ui.streamlit_app.render_chat_interface')
    def test_main_function(self, mock_render_chat, mock_render_sidebar, 
                          mock_render_header, mock_init_session, 
                          mock_setup_logging, mock_set_page_config):
        """Test main function execution."""
        from src.ui.streamlit_app import main
        
        main()
        
        # Verify all components were called
        mock_set_page_config.assert_called_once()
        mock_setup_logging.assert_called_once()
        mock_init_session.assert_called_once()
        mock_render_header.assert_called_once()
        mock_render_sidebar.assert_called_once()
        mock_render_chat.assert_called_once()
    
    def test_welcome_message_content(self):
        """Test welcome message content."""
        from src.ui.streamlit_app import get_welcome_message
        
        # Test English
        welcome_en = get_welcome_message('en')
        assert isinstance(welcome_en, str)
        assert len(welcome_en) > 0
        
        # Test Italian
        welcome_it = get_welcome_message('it')
        assert isinstance(welcome_it, str)
        assert len(welcome_it) > 0
        
        # Test fallback
        welcome_fallback = get_welcome_message('unknown')
        assert isinstance(welcome_fallback, str)
        assert len(welcome_fallback) > 0