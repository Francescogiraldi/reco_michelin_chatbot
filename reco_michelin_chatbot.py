#!/usr/bin/env python3
"""
Michelin Chatbot - Legacy Entry Point
=====================================
This file provides backward compatibility for the original interface.
For the new modular structure, use the main entry point in src/main.py

Quick Start:
- Web Interface: python -m streamlit run reco_michelin_chatbot.py
- CLI Interface: python -m src.main cli
- Full Help: python -m src.main --help
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Try to import the new modular interface
    from src.ui.streamlit_app import main
    
    if __name__ == "__main__":
        # Setup logging
        from src.utils.logging import setup_logging
        setup_logging()
        
        # Run the new Streamlit interface
        main()
        
except ImportError as e:
    # Fallback to basic interface if imports fail
    import streamlit as st
    
    st.error(
        "⚠️ **Module Import Error**\n\n"
        f"Could not import the new modular interface: {e}\n\n"
        "**Quick Fix:**\n"
        "1. Install dependencies: `pip install -r requirements.txt`\n"
        "2. Set your OpenAI API key: `export OPENAI_API_KEY=sk-...`\n"
        "3. Or create a `.env` file with: `OPENAI_API_KEY=sk-...`\n\n"
        "**Alternative:**\n"
        "Use the CLI interface: `python -m src.main cli`"
    )
    
    st.info(
        "📚 **Documentation**\n\n"
        "For detailed setup instructions, see the README.md file."
    )
