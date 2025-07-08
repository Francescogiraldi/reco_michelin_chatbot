"""Main entry point for Michelin Chatbot."""

import sys
import argparse
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import get_settings
from .services.rag_service import RAGService
from .utils.logging import setup_logging, get_logger

# Initialize console and logger
console = Console()
app = typer.Typer(help="Michelin Tire Recommendation Chatbot")


def setup_app_logging() -> None:
    """Setup application logging."""
    settings = get_settings()
    log_file = settings.logs_dir / "michelin_chatbot.log"
    setup_logging(log_file=log_file)


@app.command()
def web(
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8501, "--port", "-p", help="Port to bind to"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
) -> None:
    """Launch the web interface using Streamlit."""
    setup_app_logging()
    logger = get_logger(__name__)
    
    try:
        import streamlit.web.cli as stcli
        import streamlit as st
        
        # Get the path to the Streamlit app
        app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
        
        console.print(Panel.fit(
            f"🚀 Starting Michelin Chatbot Web Interface\n"
            f"📍 URL: http://{host}:{port}\n"
            f"🔧 Debug: {debug}",
            title="Michelin Chatbot",
            border_style="blue"
        ))
        
        # Prepare Streamlit arguments
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        if debug:
            sys.argv.extend(["--logger.level", "debug"])
        
        # Launch Streamlit
        stcli.main()
        
    except ImportError:
        console.print("❌ Streamlit not installed. Install with: pip install streamlit", style="red")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        console.print(f"❌ Failed to start web interface: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def cli(
    language: str = typer.Option("fr", "--language", "-l", help="Response language (fr, en, it, es, de)"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive mode"),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Single question to ask")
) -> None:
    """Launch the command-line interface."""
    setup_app_logging()
    logger = get_logger(__name__)
    
    try:
        console.print(Panel.fit(
            "🤖 Michelin Tire Recommendation Chatbot - CLI Mode\n"
            "Type your questions about Michelin tires and get AI-powered recommendations!",
            title="Welcome",
            border_style="green"
        ))
        
        # Initialize RAG service
        with console.status("[bold blue]Initializing AI components..."):
            rag_service = RAGService()
            catalog = rag_service.load_catalog()
            rag_service.build_vectorstore(catalog)
            rag_service.create_qa_chain(language)
        
        console.print("✅ AI components initialized successfully!", style="green")
        
        # Handle single question mode
        if question:
            _process_question(rag_service, question, language)
            return
        
        # Interactive mode
        if interactive:
            _run_interactive_cli(rag_service, language)
        
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="yellow")
    except Exception as e:
        logger.error(f"CLI error: {e}")
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


def _process_question(rag_service: RAGService, question: str, language: str) -> None:
    """Process a single question."""
    try:
        with console.status("[bold blue]Thinking..."):
            result = rag_service.query(question, language)
        
        # Display result
        console.print(Panel(
            result["result"],
            title="🤖 Recommendation",
            border_style="blue"
        ))
        
        # Display sources
        if result["source_documents"]:
            _display_sources(result["source_documents"])
        
    except Exception as e:
        console.print(f"❌ Error processing question: {e}", style="red")


def _run_interactive_cli(rag_service: RAGService, language: str) -> None:
    """Run interactive CLI mode."""
    console.print("\n💡 [bold]Tips:[/bold]")
    console.print("  • Ask about specific tire types (summer, winter, all-season)")
    console.print("  • Mention your vehicle type (car, SUV, sports car)")
    console.print("  • Type 'quit', 'exit', or press Ctrl+C to exit")
    console.print("  • Type 'help' for more information\n")
    
    while True:
        try:
            question = Prompt.ask("\n[bold blue]Your question[/bold blue]", default="")
            
            if not question.strip():
                continue
            
            if question.lower() in ['quit', 'exit', 'bye']:
                break
            
            if question.lower() == 'help':
                _show_help()
                continue
            
            _process_question(rag_service, question, language)
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    console.print("\n👋 Thank you for using Michelin Chatbot!", style="green")


def _display_sources(source_documents) -> None:
    """Display source documents in a table."""
    if not source_documents:
        return
    
    table = Table(title="📚 Source Documents", show_header=True, header_style="bold magenta")
    table.add_column("Product", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Price", style="yellow")
    table.add_column("Description", style="white")
    
    for doc in source_documents[:3]:  # Show top 3 sources
        metadata = doc.metadata
        table.add_row(
            metadata.get('name', 'Unknown'),
            metadata.get('category', 'N/A'),
            f"{metadata.get('price', 'N/A')}€",
            doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
        )
    
    console.print(table)


def _show_help() -> None:
    """Show help information."""
    help_text = """
🤖 **Michelin Chatbot Help**

**What can I help you with?**
• Find the right tires for your vehicle
• Compare different tire models
• Get recommendations based on driving conditions
• Learn about tire specifications and features

**Example questions:**
• "I need summer tires for my BMW 3 Series"
• "What are the best winter tires for safety?"
• "Recommend eco-friendly tires for city driving"
• "SUV tires for highway and light off-road use"

**Commands:**
• `quit` or `exit` - Exit the chatbot
• `help` - Show this help message
• Ctrl+C - Force exit

**Tips for better results:**
• Be specific about your vehicle type
• Mention your driving conditions (city, highway, off-road)
• Specify the season or weather conditions
• Include any special requirements (performance, eco-friendly, budget)
    """
    
    console.print(Panel(help_text, title="Help", border_style="yellow"))


@app.command()
def status() -> None:
    """Check system status and health."""
    setup_app_logging()
    logger = get_logger(__name__)
    
    try:
        console.print("🔍 Checking system status...", style="blue")
        
        # Initialize RAG service
        rag_service = RAGService()
        health = rag_service.health_check()
        
        # Create status table
        table = Table(title="System Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="yellow")
        
        for key, value in health.items():
            status_icon = "✅" if value else "❌"
            status_text = "Healthy" if value else "Issue"
            component_name = key.replace('_', ' ').title()
            
            table.add_row(component_name, f"{status_icon} {status_text}", str(value))
        
        console.print(table)
        
        # Overall status
        if health["overall_healthy"]:
            console.print("\n🎉 System is healthy and ready to use!", style="green")
        else:
            console.print("\n⚠️ System has some issues. Check the details above.", style="yellow")
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        console.print(f"❌ Status check failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def rebuild_index(
    force: bool = typer.Option(False, "--force", help="Force rebuild even if index exists")
) -> None:
    """Rebuild the vector search index."""
    setup_app_logging()
    logger = get_logger(__name__)
    
    try:
        console.print("🔨 Rebuilding vector search index...", style="blue")
        
        rag_service = RAGService()
        catalog = rag_service.load_catalog()
        rag_service.build_vectorstore(catalog, force_rebuild=force)
        
        console.print("✅ Vector index rebuilt successfully!", style="green")
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        console.print(f"❌ Index rebuild failed: {e}", style="red")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    # Handle legacy direct execution
    if len(sys.argv) == 1:
        # No arguments provided, show help
        console.print(Panel.fit(
            "🤖 Michelin Tire Recommendation Chatbot\n\n"
            "Available commands:\n"
            "• `michelin-chatbot web` - Launch web interface\n"
            "• `michelin-chatbot cli` - Launch CLI interface\n"
            "• `michelin-chatbot status` - Check system status\n"
            "• `michelin-chatbot --help` - Show detailed help",
            title="Welcome",
            border_style="blue"
        ))
        return
    
    app()


if __name__ == "__main__":
    main()