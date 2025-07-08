"""Streamlit web interface for Michelin Chatbot."""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from streamlit.runtime.caching import cache_data
from langchain.callbacks.streamlit import StreamlitCallbackHandler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import get_settings
from src.services.rag_service import RAGService
from src.models.tire import ChatMessage, ChatSession
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Michelin Tire Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .user-message {
        background-color: #f0f9ff;
        border-left-color: #0ea5e9;
    }
    
    .assistant-message {
        background-color: #f8fafc;
        border-left-color: #10b981;
    }
    
    .source-card {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-healthy { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }
</style>
""", unsafe_allow_html=True)


class StreamlitUI:
    """Streamlit user interface for Michelin Chatbot."""
    
    def __init__(self):
        """Initialize Streamlit UI."""
        self.settings = get_settings()
        self.rag_service = None
        self._initialize_session_state()
        self._initialize_rag_service()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = ChatSession(
                session_id=f"session_{int(time.time())}",
                created_at=datetime.now().isoformat()
            )
        
        if "rag_initialized" not in st.session_state:
            st.session_state.rag_initialized = False
        
        if "selected_language" not in st.session_state:
            st.session_state.selected_language = self.settings.default_language
        
        if "show_sources" not in st.session_state:
            st.session_state.show_sources = True
    
    @st.cache_resource
    def _initialize_rag_service(_self) -> RAGService:
        """Initialize RAG service with caching."""
        try:
            with st.spinner("🔧 Initializing AI components..."):
                rag_service = RAGService()
                
                # Load catalog
                catalog = rag_service.load_catalog()
                
                # Build vector store
                rag_service.build_vectorstore(catalog)
                
                # Create QA chain
                rag_service.create_qa_chain()
                
                st.session_state.rag_initialized = True
                logger.info("RAG service initialized successfully")
                
                return rag_service
                
        except Exception as e:
            st.error(f"❌ Failed to initialize AI service: {str(e)}")
            logger.error(f"RAG service initialization failed: {e}")
            st.stop()
    
    def render_header(self) -> None:
        """Render application header."""
        st.markdown("""
        <div class="main-header">
            <h1>🚗 Michelin Tire Assistant</h1>
            <p>AI-powered tire recommendations for your vehicle</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self) -> None:
        """Render sidebar with controls and information."""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Language selection
            language_options = {
                "fr": "🇫🇷 Français",
                "en": "🇬🇧 English",
                "it": "🇮🇹 Italiano",
                "es": "🇪🇸 Español",
                "de": "🇩🇪 Deutsch"
            }
            
            selected_lang = st.selectbox(
                "Response Language",
                options=list(language_options.keys()),
                format_func=lambda x: language_options[x],
                index=list(language_options.keys()).index(st.session_state.selected_language)
            )
            
            if selected_lang != st.session_state.selected_language:
                st.session_state.selected_language = selected_lang
                # Recreate QA chain with new language
                if self.rag_service:
                    self.rag_service.create_qa_chain(selected_lang)
                st.rerun()
            
            # Display options
            st.session_state.show_sources = st.checkbox(
                "Show source documents",
                value=st.session_state.show_sources
            )
            
            st.divider()
            
            # System status
            st.header("📊 System Status")
            
            if self.rag_service:
                health = self.rag_service.health_check()
                
                status_icon = "🟢" if health["overall_healthy"] else "🔴"
                st.markdown(f"{status_icon} **Overall Status**: {'Healthy' if health['overall_healthy'] else 'Issues Detected'}")
                
                with st.expander("Detailed Status"):
                    for key, value in health.items():
                        if key != "overall_healthy":
                            icon = "✅" if value else "❌"
                            st.markdown(f"{icon} {key.replace('_', ' ').title()}: {value}")
            
            st.divider()
            
            # Chat controls
            st.header("💬 Chat Controls")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_session.clear_messages()
                st.rerun()
            
            # Session info
            st.header("ℹ️ Session Info")
            st.text(f"Session ID: {st.session_state.chat_session.session_id[-8:]}")
            st.text(f"Messages: {len(st.session_state.chat_session.messages)}")
            
            if st.session_state.chat_session.created_at:
                created = datetime.fromisoformat(st.session_state.chat_session.created_at)
                st.text(f"Started: {created.strftime('%H:%M:%S')}")
    
    def render_chat_interface(self) -> None:
        """Render main chat interface."""
        # Display chat messages
        for message in st.session_state.chat_session.messages:
            self._render_message(message)
        
        # Chat input
        user_input = st.chat_input(
            "💬 Ask me about Michelin tires... (e.g., 'I need summer tires for my sports car')"
        )
        
        if user_input:
            self._handle_user_input(user_input)
    
    def _render_message(self, message: ChatMessage) -> None:
        """Render a single chat message."""
        with st.chat_message(message.role):
            st.markdown(message.content)
            
            # Show sources if available and enabled
            if (message.role == "assistant" and 
                st.session_state.show_sources and 
                message.metadata and 
                "source_documents" in message.metadata):
                
                with st.expander("📚 Source Documents", expanded=False):
                    for i, doc in enumerate(message.metadata["source_documents"]):
                        self._render_source_document(doc, i)
    
    def _render_source_document(self, doc: Any, index: int) -> None:
        """Render a source document."""
        metadata = doc.metadata
        
        st.markdown(f"""
        <div class="source-card">
            <h4>🔍 Source {index + 1}: {metadata.get('name', 'Unknown Product')}</h4>
            <p><strong>Category:</strong> {metadata.get('category', 'N/A')}</p>
            <p><strong>Price:</strong> {metadata.get('price', 'N/A')}€</p>
            <p><strong>Content:</strong> {doc.page_content[:200]}...</p>
            {f'<p><a href="{metadata.get("link", "#")}" target="_blank">🔗 View Product</a></p>' if metadata.get('link') else ''}
        </div>
        """, unsafe_allow_html=True)
    
    def _handle_user_input(self, user_input: str) -> None:
        """Handle user input and generate response."""
        # Add user message
        st.session_state.chat_session.add_message("user", user_input)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                # Create callback handler for streaming
                callback_handler = StreamlitCallbackHandler(st.container())
                
                # Query RAG service
                with st.spinner("🤔 Thinking..."):
                    result = self.rag_service.query(
                        question=user_input,
                        language=st.session_state.selected_language,
                        callbacks=[callback_handler]
                    )
                
                # Display response
                response = result["result"]
                st.markdown(response)
                
                # Add assistant message with metadata
                metadata = {
                    "source_documents": result["source_documents"],
                    "language": result["language"]
                }
                st.session_state.chat_session.add_message(
                    "assistant", 
                    response, 
                    metadata
                )
                
                # Show sources if enabled
                if st.session_state.show_sources and result["source_documents"]:
                    with st.expander("📚 Source Documents", expanded=False):
                        for i, doc in enumerate(result["source_documents"]):
                            self._render_source_document(doc, i)
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_session.add_message("assistant", error_msg)
                logger.error(f"Error processing user input: {e}")
    
    def render_welcome_message(self) -> None:
        """Render welcome message for new users."""
        if not st.session_state.chat_session.messages:
            st.info("""
            👋 **Welcome to Michelin Tire Assistant!**
            
            I'm here to help you find the perfect Michelin tires for your vehicle. You can ask me about:
            
            • 🏎️ **Performance tires** for sports cars
            • ❄️ **Winter tires** for cold weather
            • 🌦️ **All-season tires** for year-round use
            • 🚙 **SUV tires** for larger vehicles
            • 💰 **Budget-friendly options**
            • 🔧 **Technical specifications**
            
            Just type your question below and I'll provide personalized recommendations!
            """)
    
    def run(self) -> None:
        """Run the Streamlit application."""
        try:
            # Initialize RAG service
            self.rag_service = self._initialize_rag_service()
            
            # Render UI components
            self.render_header()
            self.render_sidebar()
            
            # Main content area
            col1, col2 = st.columns([3, 1])
            
            with col1:
                self.render_welcome_message()
                self.render_chat_interface()
            
            with col2:
                # Quick actions or additional info
                st.header("🚀 Quick Actions")
                
                example_questions = [
                    "I need summer tires for my BMW",
                    "Best winter tires for safety",
                    "Eco-friendly tire options",
                    "SUV tires for highway driving"
                ]
                
                st.markdown("**Try these examples:**")
                for question in example_questions:
                    if st.button(f"💡 {question}", key=f"example_{hash(question)}", use_container_width=True):
                        st.session_state.example_question = question
                        st.rerun()
                
                # Handle example question
                if hasattr(st.session_state, 'example_question'):
                    self._handle_user_input(st.session_state.example_question)
                    del st.session_state.example_question
                    st.rerun()
            
        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")
            logger.error(f"Application error: {e}")


def main() -> None:
    """Main entry point for Streamlit app."""
    try:
        app = StreamlitUI()
        app.run()
    except Exception as e:
        st.error(f"❌ Failed to start application: {str(e)}")
        logger.error(f"Failed to start application: {e}")


if __name__ == "__main__":
    main()